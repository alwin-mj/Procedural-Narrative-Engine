"""Drama manager: steers the story along a target tension curve using the viewer's affect.

Plain-language version: every scene has a *tension* (0 = calm, 1 = intense). The manager knows
how tense the story *should* be at this point (rising, peak, resolution). When a scene offers
several possible next scenes, it picks the one whose tension is closest to that goal, nudged by
how the viewer is doing:

* flat / bored (neutral face, little movement)  -> aim higher, add some excitement
* overwhelmed (very high arousal)               -> aim lower, give them a breather
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .session import Affect, StorySession

# (progress 0..1, tension 0..1): rise, peak near the end, short resolution
DEFAULT_CURVE: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.15),
    (0.4, 0.40),
    (0.85, 0.85),
    (1.0, 0.70),
)


class TensionCurve:
    """Piecewise-linear target tension as a function of story progress."""

    def __init__(self, points: Sequence[Tuple[float, float]] = DEFAULT_CURVE) -> None:
        pts = sorted(points)
        if len(pts) < 2 or pts[0][0] != 0.0 or pts[-1][0] != 1.0:
            raise ValueError("curve needs points at progress 0.0 and 1.0")
        if not all(0.0 <= t <= 1.0 for _, t in pts):
            raise ValueError("tension values must be in [0, 1]")
        self.points = tuple(pts)

    def __call__(self, progress: float) -> float:
        p = min(1.0, max(0.0, progress))
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            if p <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (p - x0) / (x1 - x0)
        return self.points[-1][1]


@dataclass(frozen=True)
class DramaConfig:
    curve: Tuple[Tuple[float, float], ...] = DEFAULT_CURVE
    flat_valence: float = 0.15  # |valence| below this AND arousal below flat_arousal = flat
    flat_arousal: float = 0.15
    overwhelmed_arousal: float = 0.7
    push: float = 0.25  # how far to shift the desired tension for flat / overwhelmed viewers


@dataclass(frozen=True)
class Decision:
    """What the director chose and why. ``to_id is None`` means 'go to the ending'."""

    from_id: str
    to_id: Optional[str]
    mood: str
    state: str  # "flat", "engaged" or "overwhelmed"
    target: float  # tension the curve asks for
    desired: float  # target after adjusting for the viewer
    options: Tuple[Tuple[str, float], ...]  # (scene id, tension) of each allowed next scene
    reason: str


class DramaManager:
    def __init__(self, session: StorySession, cfg: Optional[DramaConfig] = None) -> None:
        self.session = session
        self.cfg = cfg = cfg or DramaConfig()
        self.curve = TensionCurve(cfg.curve)
        self._steps_total = max(1, session.graph.longest_path() - 1)

    def viewer_state(self, affect: Affect) -> str:
        c = self.cfg
        if affect.arousal > c.overwhelmed_arousal:
            return "overwhelmed"
        if abs(affect.valence) < c.flat_valence and affect.arousal < c.flat_arousal:
            return "flat"
        return "engaged"

    def progress_after_next_step(self) -> float:
        """Story progress (0..1) at the scene we are about to enter."""
        return min(1.0, len(self.session.state.visited) / self._steps_total)

    def decide(self, mood: str, affect: Affect) -> Decision:
        s = self.session
        here = s.node
        if here.terminal:
            reason = "final scene played -> ending"
            return Decision(here.id, None, mood, "n/a", 0.0, 0.0, (), reason)

        state = self.viewer_state(affect)
        target = self.curve(self.progress_after_next_step())
        shift = {"flat": +self.cfg.push, "overwhelmed": -self.cfg.push}.get(state, 0.0)
        desired = min(1.0, max(0.0, target + shift))

        edges = s.candidates(mood, affect)
        options = tuple((e.to, s.graph.nodes[e.to].tension) for e in edges)
        # closest tension to the goal; ties go to the edge listed first in the story file
        best = min(range(len(options)), key=lambda i: (abs(options[i][1] - desired), i))
        to_id, tension = options[best]

        why = {
            "flat": f"viewer flat (V{affect.valence:+.2f} A{affect.arousal:.2f}) -> aim higher",
            "overwhelmed": f"viewer overwhelmed (A{affect.arousal:.2f}) -> aim lower",
            "engaged": "viewer engaged -> follow the curve",
        }[state]
        others = ", ".join(f"{i}({t:.2f})" for i, t in options if i != to_id)
        reason = f"{why}; want tension {desired:.2f}; {here.id}->{to_id} ({tension:.2f})" + (
            f" over {others}" if others else ""
        )
        return Decision(here.id, to_id, mood, state, target, desired, options, reason)

    def decide_fixed(self, mood: str, affect: Affect) -> Decision:
        """Baseline for evaluation: always the first allowed edge, ignoring tension and affect."""
        s = self.session
        here = s.node
        if here.terminal:
            return Decision(here.id, None, mood, "n/a", 0.0, 0.0, (), "final scene -> ending")
        edges = s.candidates(mood, affect)
        to_id = edges[0].to
        options = tuple((e.to, s.graph.nodes[e.to].tension) for e in edges)
        return Decision(
            here.id, to_id, mood, "n/a", 0.0, 0.0, options, f"fixed order: {here.id}->{to_id}"
        )

    def decide_random(self, mood: str, affect: Affect, rng: random.Random) -> Decision:
        """Baseline for evaluation: a random allowed edge."""
        s = self.session
        here = s.node
        if here.terminal:
            return Decision(here.id, None, mood, "n/a", 0.0, 0.0, (), "final scene -> ending")
        edges = s.candidates(mood, affect)
        to_id = rng.choice(edges).to
        options = tuple((e.to, s.graph.nodes[e.to].tension) for e in edges)
        return Decision(
            here.id, to_id, mood, "n/a", 0.0, 0.0, options, f"random: {here.id}->{to_id}"
        )
