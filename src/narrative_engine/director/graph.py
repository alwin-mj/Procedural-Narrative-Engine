"""Story graph: scenes as nodes, edges with conditions, endings chosen by trajectory.

A story file (JSON) looks like::

    {
      "title": "Arin's Day",
      "protagonist": "Arin",
      "start": "wake",
      "scenes": [
        {
          "id": "smithy",
          "location": "river smithy",
          "tension": 0.45,
          "variants": {"neutral": "...", "happy": "...", "uplift": "...", "sad": "..."},
          "effects":   [{"when": {"mood": "happy"}, "set_flags": ["pendant"],
                         "relations": {"smith": 1}}],
          "callbacks": [{"if": {"flag": "map"}, "text": "The guild map marks this forge."}],
          "next": [{"to": "cottage"}, {"to": "bridge", "when": {"arousal": ">=0.3"}}]
        }
      ],
      "endings": [
        {"id": "radiant", "when": {"avg_valence": ">=0.25"}, "text": "..."},
        {"id": "wanderer", "text": "..."}
      ]
    }

Condition keys (all must hold): ``mood``, ``valence``, ``arousal``, ``avg_valence``,
``avg_arousal`` (comparators such as ``">=0.3"``, or a list for a range), ``flag``, ``not_flag``,
``visited``, ``not_visited``, ``relation`` (``{"name": ">=1"}``).

A scene with no ``next`` is terminal and leads to an ending. The older linear format
(``scenes`` + a single ``ending`` string) is still accepted and becomes a chain.
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from ..config import MOODS

DEFAULT_STORY = Path(__file__).resolve().parent.parent / "data" / "stories" / "arin_adventure.json"

_CMP = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
NUMERIC_KEYS = ("valence", "arousal", "avg_valence", "avg_arousal")
LIST_KEYS = ("mood", "flag", "not_flag", "visited", "not_visited")
CONDITION_KEYS = NUMERIC_KEYS + LIST_KEYS + ("relation",)


class StoryError(ValueError):
    """Raised when story data is malformed."""


# ---------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Context:
    """Everything a condition can look at."""

    mood: str = "neutral"
    valence: float = 0.0  # recent affect (mean over the scene just played)
    arousal: float = 0.0
    avg_valence: float = 0.0  # affect averaged over the whole playthrough so far
    avg_arousal: float = 0.0
    flags: FrozenSet[str] = frozenset()
    visited: Tuple[str, ...] = ()
    relations: Mapping[str, int] = field(default_factory=dict)


def _as_list(value: Any) -> List[Any]:
    return [value] if isinstance(value, str) else list(value)


def _compare(value: float, expr: str) -> bool:
    m = _CMP.match(expr)
    if m is None:  # validated at load time; defensive for hand-built conditions
        raise StoryError(f"bad comparator {expr!r}")
    return bool(_OPS[m.group(1)](value, float(m.group(2))))


@dataclass(frozen=True)
class Condition:
    """A conjunction of tests. An empty condition is always true."""

    spec: Mapping[str, Any] = field(default_factory=dict)

    @property
    def always(self) -> bool:
        return not self.spec

    def test(self, ctx: Context) -> bool:
        for key, want in self.spec.items():
            if key == "mood":
                ok = ctx.mood in _as_list(want)
            elif key in NUMERIC_KEYS:
                ok = all(_compare(getattr(ctx, key), e) for e in _as_list(want))
            elif key == "flag":
                ok = all(f in ctx.flags for f in _as_list(want))
            elif key == "not_flag":
                ok = not any(f in ctx.flags for f in _as_list(want))
            elif key == "visited":
                ok = all(v in ctx.visited for v in _as_list(want))
            elif key == "not_visited":
                ok = not any(v in ctx.visited for v in _as_list(want))
            else:  # relation
                ok = all(_compare(float(ctx.relations.get(n, 0)), e) for n, e in want.items())
            if not ok:
                return False
        return True


def parse_condition(spec: Any, where: str) -> Condition:
    """Validate a raw condition dict and wrap it."""
    if spec is None:
        return Condition()
    if not isinstance(spec, Mapping):
        raise StoryError(f"{where}: condition must be an object")
    for key, want in spec.items():
        if key not in CONDITION_KEYS:
            raise StoryError(f"{where}: unknown condition key {key!r}")
        if key in NUMERIC_KEYS:
            for expr in _as_list(want) if isinstance(want, (str, list, tuple)) else [want]:
                if not isinstance(expr, str) or not _CMP.match(expr):
                    raise StoryError(f"{where}: {key} needs comparators like '>=0.3', got {expr!r}")
        elif key == "mood":
            for m in _as_list(want) if isinstance(want, (str, list, tuple)) else [want]:
                if m not in MOODS:
                    raise StoryError(f"{where}: unknown mood {m!r}")
        elif key == "relation":
            if not isinstance(want, Mapping) or not all(
                isinstance(e, str) and _CMP.match(e) for e in want.values()
            ):
                raise StoryError(f"{where}: relation needs {{name: comparator}}")
        else:
            items = _as_list(want) if isinstance(want, (str, list, tuple)) else [want]
            if not items or not all(isinstance(i, str) and i for i in items):
                raise StoryError(f"{where}: {key} needs a name or a list of names")
    return Condition(dict(spec))


# ---------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Edge:
    to: str
    when: Condition = field(default_factory=Condition)


@dataclass(frozen=True)
class Effect:
    when: Condition = field(default_factory=Condition)
    set_flags: Tuple[str, ...] = ()
    relations: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Callback:
    cond: Condition
    text: str


@dataclass(frozen=True)
class Node:
    id: str
    variants: Mapping[str, str]
    tension: float = 0.5
    location: str = ""
    effects: Tuple[Effect, ...] = ()
    callbacks: Tuple[Callback, ...] = ()
    edges: Tuple[Edge, ...] = ()

    @property
    def terminal(self) -> bool:
        return not self.edges


@dataclass(frozen=True)
class Ending:
    id: str
    text: str
    when: Condition = field(default_factory=Condition)


class StoryGraph:
    """Validated, immutable story data."""

    def __init__(
        self,
        nodes: List[Node],
        endings: List[Ending],
        start: Optional[str] = None,
        title: str = "",
        protagonist: str = "",
    ) -> None:
        self.nodes: Dict[str, Node] = {}
        for n in nodes:
            if n.id in self.nodes:
                raise StoryError(f"duplicate scene id: {n.id!r}")
            self.nodes[n.id] = n
        self.endings = endings
        self.start = start or (nodes[0].id if nodes else "")
        self.title = title
        self.protagonist = protagonist
        self._validate()
        self._longest: Dict[str, int] = {}
        self._shortest: Dict[str, int] = {}

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        if not self.nodes:
            raise StoryError("story needs at least one scene")
        if self.start not in self.nodes:
            raise StoryError(f"start scene {self.start!r} does not exist")

        for n in self.nodes.values():
            for mood in MOODS:
                text = n.variants.get(mood)
                if not isinstance(text, str) or not text.strip():
                    raise StoryError(f"scene {n.id!r} is missing a line for mood {mood!r}")
            if not 0.0 <= n.tension <= 1.0:
                raise StoryError(f"scene {n.id!r}: tension must be in [0, 1]")
            for e in n.edges:
                if e.to not in self.nodes:
                    raise StoryError(f"scene {n.id!r}: edge to unknown scene {e.to!r}")
                if e.to == n.id:
                    raise StoryError(f"scene {n.id!r}: edge to itself")
            if n.edges and not any(e.when.always for e in n.edges):
                raise StoryError(
                    f"scene {n.id!r}: needs at least one edge without 'when' (dead-end risk)"
                )
            for cb in n.callbacks:
                if not cb.text.strip():
                    raise StoryError(f"scene {n.id!r}: empty callback text")

        # reachability
        seen, stack = set(), [self.start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(e.to for e in self.nodes[cur].edges)
        unreachable = sorted(set(self.nodes) - seen)
        if unreachable:
            raise StoryError(f"unreachable scenes: {unreachable}")

        # acyclic (guarantees every playthrough terminates)
        state: Dict[str, int] = {}

        def visit(nid: str) -> None:
            if state.get(nid) == 1:
                raise StoryError(f"cycle detected at scene {nid!r}")
            if state.get(nid) == 2:
                return
            state[nid] = 1
            for e in self.nodes[nid].edges:
                visit(e.to)
            state[nid] = 2

        visit(self.start)

        # endings
        if not self.endings:
            raise StoryError("story needs at least one ending")
        ids = [e.id for e in self.endings]
        if len(set(ids)) != len(ids):
            raise StoryError("duplicate ending ids")
        for e in self.endings:
            if not e.text.strip():
                raise StoryError(f"ending {e.id!r} has no text")
        if not self.endings[-1].when.always:
            raise StoryError("the last ending must have no 'when' (it is the default)")

    # ------------------------------------------------------------------ queries
    def longest_path(self, nid: Optional[str] = None) -> int:
        """Number of scenes on the longest route from ``nid`` (default: start) to an ending."""
        nid = nid or self.start
        if nid not in self._longest:
            n = self.nodes[nid]
            self._longest[nid] = 1 + max((self.longest_path(e.to) for e in n.edges), default=0)
        return self._longest[nid]

    def shortest_path(self, nid: Optional[str] = None) -> int:
        nid = nid or self.start
        if nid not in self._shortest:
            n = self.nodes[nid]
            self._shortest[nid] = 1 + min((self.shortest_path(e.to) for e in n.edges), default=0)
        return self._shortest[nid]

    @property
    def terminals(self) -> List[str]:
        return [n.id for n in self.nodes.values() if n.terminal]

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StoryGraph":
        if not isinstance(data, Mapping):
            raise StoryError("story must be a JSON object")
        raw_scenes = data.get("scenes")
        if not isinstance(raw_scenes, list):
            raise StoryError("story must contain a 'scenes' list")

        legacy = "endings" not in data and "ending" in data
        nodes: List[Node] = []
        for i, raw in enumerate(raw_scenes):
            where = f"scene #{i}"
            if not isinstance(raw, Mapping) or "variants" not in raw:
                raise StoryError(f"{where} must be an object with 'variants'")
            nid = str(raw.get("id", f"scene_{i}"))
            where = f"scene {nid!r}"
            if legacy:  # old linear format: chain scenes in order
                edges: Tuple[Edge, ...] = ()
                if i + 1 < len(raw_scenes):
                    nxt = raw_scenes[i + 1]
                    edges = (Edge(str(nxt.get("id", f"scene_{i + 1}"))),)
            else:
                edges = tuple(_parse_edge(e, where) for e in raw.get("next", []) or [])
            try:
                tension = float(raw.get("tension", 0.5))
            except (TypeError, ValueError) as exc:
                raise StoryError(f"{where}: tension must be a number") from exc
            nodes.append(
                Node(
                    id=nid,
                    variants=dict(raw["variants"]),
                    tension=tension,
                    location=str(raw.get("location", "")),
                    effects=tuple(_parse_effect(e, where) for e in raw.get("effects", []) or []),
                    callbacks=tuple(
                        _parse_callback(c, where) for c in raw.get("callbacks", []) or []
                    ),
                    edges=edges,
                )
            )

        if legacy:
            endings = [Ending("end", str(data["ending"]))]
        else:
            endings = [_parse_ending(e, i) for i, e in enumerate(data.get("endings", []) or [])]
        return cls(
            nodes,
            endings,
            start=data.get("start"),
            title=str(data.get("title", "")),
            protagonist=str(data.get("protagonist", "")),
        )

    @classmethod
    def from_file(cls, path: Path) -> "StoryGraph":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise StoryError(f"cannot read story file {path}: {exc}") from exc
        return cls.from_dict(data)


def _parse_edge(raw: Any, where: str) -> Edge:
    if isinstance(raw, str):
        return Edge(raw)
    if not isinstance(raw, Mapping) or "to" not in raw:
        raise StoryError(f"{where}: each 'next' entry needs a 'to'")
    return Edge(str(raw["to"]), parse_condition(raw.get("when"), f"{where} edge->{raw['to']}"))


def _parse_effect(raw: Any, where: str) -> Effect:
    if not isinstance(raw, Mapping):
        raise StoryError(f"{where}: effect must be an object")
    flags = raw.get("set_flags", []) or []
    rels = raw.get("relations", {}) or {}
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise StoryError(f"{where}: set_flags must be a list of names")
    if not isinstance(rels, Mapping) or not all(isinstance(v, int) for v in rels.values()):
        raise StoryError(f"{where}: relations must map names to integers")
    return Effect(parse_condition(raw.get("when"), f"{where} effect"), tuple(flags), dict(rels))


def _parse_callback(raw: Any, where: str) -> Callback:
    if not isinstance(raw, Mapping) or "text" not in raw:
        raise StoryError(f"{where}: callback needs 'text'")
    return Callback(parse_condition(raw.get("if"), f"{where} callback"), str(raw["text"]))


def _parse_ending(raw: Any, i: int) -> Ending:
    if not isinstance(raw, Mapping) or "text" not in raw:
        raise StoryError(f"ending #{i} needs 'text'")
    eid = str(raw.get("id", f"ending_{i}"))
    return Ending(eid, str(raw["text"]), parse_condition(raw.get("when"), f"ending {eid!r}"))


def load_story(path: Optional[Path] = None) -> StoryGraph:
    """Load ``path`` or the bundled default story."""
    return StoryGraph.from_file(path or DEFAULT_STORY)
