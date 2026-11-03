"""Cheap whole-frame motion test used to speed up the on-screen character."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


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
