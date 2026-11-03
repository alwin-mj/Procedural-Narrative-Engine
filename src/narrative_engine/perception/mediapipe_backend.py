"""MediaPipe Face Landmarker backend: fast CPU face tracking + 52 blendshape scores.

The model file (``face_landmarker.task``, ~4 MB) is downloaded once from Google's public
MediaPipe model bucket and cached in ``~/.cache/narrative-engine``. This is the only network
access the project ever makes; pass ``--mp-model PATH`` to use a local copy and stay fully
offline.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_NAME = "face_landmarker.task"
MIN_MODEL_BYTES = 1_000_000


class ModelUnavailable(RuntimeError):
    """The Face Landmarker model file could not be found or downloaded."""


class FaceObservation(NamedTuple):
    box: Box
    blendshapes: Dict[str, float]


def default_model_path() -> Path:
    return Path.home() / ".cache" / "narrative-engine" / MODEL_NAME


def ensure_model(
    path: Optional[Union[str, Path]] = None, *, url: str = MODEL_URL, timeout: float = 30.0
) -> Path:
    """Return a usable model path, downloading to the cache on first use."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ModelUnavailable(f"MediaPipe model file not found: {p}")
        return p

    p = default_model_path()
    if p.is_file() and p.stat().st_size >= MIN_MODEL_BYTES:
        return p

    log.info("Downloading MediaPipe face model (one time, ~4 MB)...")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except (OSError, urllib.error.URLError) as exc:  # URLError is an OSError subclass
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(
            f"could not download the MediaPipe model ({exc}). Download it manually from "
            f"{url} and pass --mp-model PATH."
        ) from exc
    if tmp.stat().st_size < MIN_MODEL_BYTES:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable("downloaded MediaPipe model is incomplete")
    tmp.replace(p)
    return p


class MediaPipeBackend:
    """Thin wrapper over ``mediapipe.tasks.vision.FaceLandmarker`` (VIDEO mode)."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        import mediapipe as mp  # optional dependency; ImportError is handled by the caller
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def observe(self, rgb: np.ndarray) -> Optional[FaceObservation]:
        """Face box + blendshape scores for an RGB frame, or ``None`` if no face is visible."""
        h, w = rgb.shape[:2]
        data = np.ascontiguousarray(rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=data)
        ts = max(self._last_ts_ms + 1, int(time.monotonic() * 1000))  # must strictly increase
        self._last_ts_ms = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None

        pts = result.face_landmarks[0]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        x0, y0 = max(0, int(min(xs) * w)), max(0, int(min(ys) * h))
        x1, y1 = min(w, int(max(xs) * w)), min(h, int(max(ys) * h))
        blend: Dict[str, float] = {}
        if result.face_blendshapes:
            blend = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        return FaceObservation((x0, y0, max(1, x1 - x0), max(1, y1 - y0)), blend)

    def close(self) -> None:
        self._landmarker.close()
