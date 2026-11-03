"""Cheap whole-frame motion test used to speed up the on-screen character."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


class MotionDetector:
    """Reports ``True`` when the mean absolute frame difference exceeds a threshold."""

    def __init__(self, threshold: float = 7.0) -> None:
        self.threshold = threshold
        self.last_motion = 0.0
        self._prev_gray: Optional[np.ndarray] = None

    def update(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        moving = False
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self.last_motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            moving = self.last_motion > self.threshold
        self._prev_gray = gray
        return moving


class FaceMotion:
    """Moving-viewer test based on how far the face moves between frames.

    Speed is the face-centre displacement divided by face width, so it does not depend on camera
    resolution or how far you sit from the camera. A rolling median ignores one-frame jitter, and
    global lighting changes (which fool whole-frame differencing) have no effect.
    """

    def __init__(self, threshold: float = 0.03, window: int = 5, min_samples: int = 3) -> None:
        self.threshold = threshold
        self.min_samples = min_samples
        self.last_motion = 0.0
        self._speeds: Deque[float] = deque(maxlen=window)
        self._prev: Optional[Tuple[float, float]] = None

    def update(self, box: Optional[Box]) -> bool:
        if box is None:
            self._prev = None
            self._speeds.clear()
            self.last_motion = 0.0
            return False
        x, y, w, h = box
        centre = (x + w / 2.0, y + h / 2.0)
        if self._prev is not None:
            dist = math.hypot(centre[0] - self._prev[0], centre[1] - self._prev[1])
            self._speeds.append(dist / max(1.0, float(w)))
        self._prev = centre
        if len(self._speeds) < self.min_samples:
            return False
        self.last_motion = float(np.median(self._speeds))
        return self.last_motion > self.threshold
