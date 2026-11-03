"""Temporal smoothing so the mood does not flicker frame to frame."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Optional, Tuple

from ..config import MOODS

Scores = Dict[str, float]


class EmotionSmoother:
    """Rolling average over recent per-frame mood distributions.

    The stable mood is the highest averaged score, with a bias towards "neutral"
    unless a non-neutral mood is sustained or close to neutral.
    """

    def __init__(self, buffer_size: int = 10, window: int = 6) -> None:
        if window > buffer_size:
            raise ValueError("window cannot exceed buffer_size")
        self._buffer: Deque[Mapping[str, float]] = deque(maxlen=buffer_size)
        self._window = window

    def reset(self) -> None:
        self._buffer.clear()

    def update(
        self, frame_scores: Mapping[str, float], dominant: str, confidence: float
    ) -> Tuple[str, float, Scores]:
        """Add one frame and return ``(stable_mood, stable_confidence, averaged_scores)``."""
        self._buffer.append(frame_scores)
        window = list(self._buffer)[-self._window :]

        avg: Scores = {m: 0.0 for m in MOODS}
        counts = {m: 0 for m in MOODS}
        for d in window:
            for m in MOODS:
                v = float(d.get(m, 0.0))
                avg[m] += v
                if v > 0.01:
                    counts[m] += 1
        for m in MOODS:
            avg[m] /= len(window)

        stable = max(avg, key=avg.get)  # type: ignore[arg-type]
        stable_avg = avg[stable]
        neutral = avg["neutral"]

        if stable != "neutral":
            sustained = counts[stable] >= 2 and stable_avg >= 0.40
            close_to_neutral = stable_avg >= neutral * 0.7
            if not (sustained or close_to_neutral):
                if neutral >= 0.2:
                    stable, stable_avg = "neutral", neutral
                else:
                    stable, stable_avg = dominant, confidence
        else:
            stable_avg = avg.get("neutral", confidence)

        return stable, stable_avg, avg


class Ema:
    """Exponential moving average: ``value = alpha * x + (1 - alpha) * value``."""

    def __init__(self, alpha: float = 0.3, initial: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = initial
        self._initial = initial

    def update(self, x: float) -> float:
        self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = self._initial


class HysteresisGate:
    """Stops the mood flipping on small differences.

    The held mood only changes when a different candidate beats it by ``margin`` (in averaged
    score) for ``hold_frames`` consecutive frames.
    """

    def __init__(self, margin: float = 0.08, hold_frames: int = 3) -> None:
        if margin < 0 or hold_frames < 1:
            raise ValueError("margin must be >= 0 and hold_frames >= 1")
        self.margin = margin
        self.hold_frames = hold_frames
        self.current: Optional[str] = None
        self._pending: Optional[str] = None
        self._count = 0

    def reset(self) -> None:
        self.current = None
        self._pending = None
        self._count = 0

    def update(self, candidate: str, avg: Mapping[str, float]) -> str:
        if self.current is None:
            self.current = candidate
            return candidate
        if candidate == self.current:
            self._pending, self._count = None, 0
            return self.current
        if avg.get(candidate, 0.0) - avg.get(self.current, 0.0) < self.margin:
            self._pending, self._count = None, 0
            return self.current
        if candidate == self._pending:
            self._count += 1
        else:
            self._pending, self._count = candidate, 1
        if self._count >= self.hold_frames:
            self.current = candidate
            self._pending, self._count = None, 0
        return self.current
