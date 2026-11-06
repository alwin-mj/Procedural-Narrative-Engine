"""A playthrough of a story graph: current scene, story state, and trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set

from .graph import Context, Edge, Node, StoryError, StoryGraph


class Affect(NamedTuple):
    """Mean valence/arousal over a stretch of play."""

    valence: float = 0.0
    arousal: float = 0.0


@dataclass
class StoryState:
    """Everything that carries over between scenes."""

    flags: Set[str] = field(default_factory=set)  # inventory items, facts learned
    relations: Dict[str, int] = field(default_factory=dict)  # character -> affinity
    visited: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)  # mood each scene was played in
    v_sum: float = 0.0
    a_sum: float = 0.0
    n_scenes: int = 0  # scenes whose affect has been recorded

    def copy(self) -> "StoryState":
        return StoryState(
            set(self.flags),
            dict(self.relations),
            list(self.visited),
            list(self.moods),
            self.v_sum,
            self.a_sum,
            self.n_scenes,
        )


class StorySession:
    """Plays through a :class:`StoryGraph`. Contains no camera or timing logic."""

    def __init__(self, graph: StoryGraph) -> None:
        self.graph = graph
        self.state = StoryState()
        self.current_id = graph.start
        self.finished = False
        self.ending_id: Optional[str] = None
        self._last_affect = Affect()
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.state = StoryState()
        self.current_id = self.graph.start
        self.finished = False
        self.ending_id = None
        self._last_affect = Affect()
        self._enter(self.state, self.graph.nodes[self.graph.start], "neutral", Affect())

    @property
    def node(self) -> Node:
        return self.graph.nodes[self.current_id]

    @property
    def step_count(self) -> int:
        """How many times the story has advanced since the start."""
        return len(self.state.visited) - 1

    @property
    def is_terminal(self) -> bool:
        return self.node.terminal

    # ------------------------------------------------------------------ context
    def context(
        self,
        mood: str,
        affect: Optional[Affect] = None,
        state: Optional[StoryState] = None,
        pending: Optional[Affect] = None,
    ) -> Context:
        """Build the context conditions are tested against.

        ``pending`` adds a scene that has been played but not yet recorded, so the averages
        include it.
        """
        affect = affect or Affect()
        st = state or self.state
        v_sum, a_sum, n = st.v_sum, st.a_sum, st.n_scenes
        if pending is not None:
            v_sum, a_sum, n = v_sum + pending.valence, a_sum + pending.arousal, n + 1
        return Context(
            mood=mood,
            valence=affect.valence,
            arousal=affect.arousal,
            avg_valence=v_sum / n if n else 0.0,
            avg_arousal=a_sum / n if n else 0.0,
            flags=frozenset(st.flags),
            visited=tuple(st.visited),
            relations=dict(st.relations),
        )

    # ------------------------------------------------------------------ text
    @staticmethod
    def _compose(node: Node, mood: str, ctx: Context) -> str:
        parts = [node.variants[mood]]
        parts.extend(cb.text for cb in node.callbacks if cb.cond.test(ctx))
        return " ".join(parts)

    def get_line(self, mood: str) -> str:
        """Text for the current scene (or the chosen ending once the story is over)."""
        if self.finished:
            return self.ending_text()
        ctx = self.context(mood, self._last_affect)
        return self._compose(self.node, mood, ctx)

    def preview_line(self, to_id: str, mood: str, affect: Affect) -> str:
        """The text scene ``to_id`` would show if the story moved there now."""
        node = self.graph.nodes[to_id]
        st = self.state.copy()
        self._record(st, affect)
        self._enter(st, node, mood, affect)
        return self._compose(node, mood, self.context(mood, affect, state=st))

    def ending_text(self) -> str:
        for e in self.graph.endings:
            if e.id == self.ending_id:
                return e.text
        return self.graph.endings[-1].text

    # ------------------------------------------------------------------ transitions
    def candidates(self, mood: str, affect: Affect) -> List[Edge]:
        """Edges that may be taken now (never empty for a non-terminal scene)."""
        ctx = self.context(mood, affect, pending=affect)
        return [e for e in self.node.edges if e.when.test(ctx)]

    def advance(self, to_id: str, mood: str, affect: Affect) -> str:
        """Leave the current scene (recording its mean ``affect``) and enter ``to_id``."""
        if self.finished:
            raise StoryError("story is finished")
        if to_id not in {e.to for e in self.node.edges}:
            raise StoryError(f"no edge from {self.current_id!r} to {to_id!r}")
        self._record(self.state, affect)
        self.current_id = to_id
        self._last_affect = affect
        self._enter(self.state, self.node, mood, affect)
        return self.get_line(mood)

    def finish(self, mood: str, affect: Affect) -> str:
        """Leave the final scene and pick the ending that matches the whole trajectory."""
        if self.finished:
            return self.ending_text()
        if not self.is_terminal:
            raise StoryError(f"scene {self.current_id!r} is not terminal")
        self._record(self.state, affect)
        ctx = self.context(mood, affect)
        for e in self.graph.endings:
            if e.when.test(ctx):
                self.ending_id = e.id
                break
        self.finished = True
        return self.ending_text()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _record(state: StoryState, affect: Affect) -> None:
        state.v_sum += affect.valence
        state.a_sum += affect.arousal
        state.n_scenes += 1

    def _enter(self, state: StoryState, node: Node, mood: str, affect: Affect) -> None:
        state.visited.append(node.id)
        state.moods.append(mood)
        ctx = self.context(mood, affect, state=state)
        for effect in node.effects:
            if effect.when.test(ctx):
                state.flags.update(effect.set_flags)
                for name, delta in effect.relations.items():
                    state.relations[name] = state.relations.get(name, 0) + delta
