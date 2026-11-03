"""Temporal smoothing so the mood does not flicker frame to frame."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Tuple

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
