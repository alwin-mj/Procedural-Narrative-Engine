"""Validate a story file and draw its graph.

narrative-story check my_story.json        # validate + print a summary
narrative-story mermaid my_story.json      # print a Mermaid flowchart (renders on GitHub)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .director import StoryError, StoryGraph, load_story


def describe_condition(spec: Mapping[str, Any]) -> str:
    """Short human-readable form of a condition, e.g. ``arousal >=0.3, mood happy``."""
    parts: List[str] = []
    for key, want in spec.items():
        if key == "relation":
            parts.append(", ".join(f"{n} {e}" for n, e in want.items()))
        elif isinstance(want, (list, tuple)):
            parts.append(f"{key} " + " & ".join(str(w) for w in want))
        else:
            parts.append(f"{key} {want}")
    return ", ".join(parts)


def to_mermaid(graph: StoryGraph) -> str:
    """Mermaid flowchart: nodes show tension, conditional edges show their condition."""
    lines = ["flowchart TD"]
    for n in graph.nodes.values():
        lines.append(f'    {n.id}["{n.id}<br/>tension {n.tension:.2f}"]')
    for n in graph.nodes.values():
        for e in n.edges:
            label = describe_condition(e.when.spec) if not e.when.always else ""
            arrow = f'-->|"{label}"|' if label else "-->"
            lines.append(f"    {n.id} {arrow} {e.to}")
    for end in graph.endings:
        label = describe_condition(end.when.spec) if not end.when.always else "default"
        lines.append(f'    END_{end.id}(["ending: {end.id}<br/>{label}"])')
        for t in graph.terminals:
            lines.append(f"    {t} --> END_{end.id}")
    return "\n".join(lines)


def summary(graph: StoryGraph) -> str:
    conditional = sum(1 for n in graph.nodes.values() for e in n.edges if not e.when.always)
    return "\n".join(
        [
            f"OK: {graph.title or 'untitled'}",
            f"  scenes: {len(graph.nodes)} (start: {graph.start}, terminal: {graph.terminals})",
            f"  routes: shortest {graph.shortest_path()} scenes, longest {graph.longest_path()}",
            f"  conditional edges: {conditional}",
            f"  endings: {[e.id for e in graph.endings]}",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="narrative-story", description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("check", "mermaid"))
    p.add_argument("file", type=Path, nargs="?", help="story JSON (default: bundled story)")
    args = p.parse_args(argv)
    try:
        graph = load_story(args.file)
    except StoryError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(summary(graph) if args.command == "check" else to_mermaid(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
