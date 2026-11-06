"""Optional sprite animation loaded from a folder of PNG frames."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class PlayerAnimation:
    """Loads frames in alphabetical order, resizes them, and cycles at ``fps``."""

    def __init__(
        self,
        folder_path: Optional[Union[str, Path]],
        size: Tuple[int, int] = (160, 160),
        fps: int = 10,
    ) -> None:
        self.frames: List[np.ndarray] = []
        self.idx = 0
        self.last_time = 0.0
        self.fps = fps
        self.size = size

        if folder_path and os.path.isdir(folder_path):
            for fn in sorted(os.listdir(folder_path)):
                if not fn.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                img = cv2.imread(os.path.join(folder_path, fn), cv2.IMREAD_UNCHANGED)
                if img is None:
                    log.warning("Skipping unreadable sprite frame: %s", fn)
                    continue
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.frames.append(img)
        elif folder_path:
            log.warning("Sprite folder not found: %s (using the default orb)", folder_path)

        self.n = len(self.frames)
        self.last_time = time.monotonic()  # start the animation clock once loading is done

    @property
    def has_frames(self) -> bool:
        return self.n > 0

    def get_frame(self) -> Optional[np.ndarray]:
        if self.n == 0:
            return None
        now = time.monotonic()
        if now - self.last_time >= 1.0 / self.fps:
            self.idx = (self.idx + 1) % self.n
            self.last_time = now
        return self.frames[self.idx]

    @staticmethod
    def overlay_png(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
        """Alpha-blend ``fg`` (BGRA or BGR) onto ``bg`` with its top-left at ``(x, y)``."""
        h, w = fg.shape[:2]
        bh, bw = bg.shape[:2]
        if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
            return bg
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bw, x + w), min(bh, y + h)
        fx1, fy1 = x1 - x, y1 - y
        fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

        bg_region = bg[y1:y2, x1:x2].astype(np.float32)
        fg_region = fg[fy1:fy2, fx1:fx2].astype(np.float32)

        if fg_region.shape[2] == 4:
            alpha = fg_region[:, :, 3:4] / 255.0
            blended = alpha * fg_region[:, :, :3] + (1 - alpha) * bg_region
        else:
            blended = 0.9 * fg_region[:, :, :3] + 0.1 * bg_region
        bg[y1:y2, x1:x2] = blended.astype(bg.dtype)
        return bg
