"""Face + emotion detection with an optional-backend fallback chain.

Chain (``backend="auto"``): FER -> DeepFace -> Haar face + smile heuristic.
FER and DeepFace are optional installs; if they are missing the next one takes over,
so the app always starts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from ..config import MOODS
from .smoothing import EmotionSmoother, Scores

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x, y, w, h

# Raw label -> one of the four story moods.
EMOTION_MAP: Dict[str, str] = {
    "happy": "happy",
    "content": "happy",
    "joy": "happy",
    "happiness": "happy",
    "surprise": "uplift",
    "surprised": "uplift",
    "excited": "uplift",
    "shocked": "uplift",
    "neutral": "neutral",
    "calm": "neutral",
    "bored": "neutral",
    "confused": "neutral",
    "serious": "neutral",
    "tired": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "angry": "sad",
    "anger": "sad",
    "fear": "sad",
    "fearful": "sad",
    "disgust": "sad",
    "disgusted": "sad",
    "embarrassed": "sad",
    "worried": "sad",
}

FACE_PAD = 0.12  # padding around the detected face box
SMILE_NUDGE_FER = 0.70  # FER: nudge neutral -> happy when confidence is below this
SMILE_NUDGE_DEEPFACE = 0.65
SMILE_SCORE = 0.55  # happy score assigned when the smile cue fires


class Detection(NamedTuple):
    """Result of one detection step."""

    emotion: str
    confidence: float
    box: Optional[Box]
    scores: Scores


def map_emotion(raw: Any) -> str:
    """Fold a raw model label into one of the four story moods (unknown -> neutral)."""
    return EMOTION_MAP.get(str(raw).lower(), "neutral")


def _empty_scores(neutral: float = 0.0) -> Scores:
    s: Scores = {m: 0.0 for m in MOODS}
    s["neutral"] = neutral
    return s


def _fold_scores(raw: Mapping[str, float], scale: float = 1.0) -> Scores:
    """Sum raw per-label scores into the four moods and normalise to sum 1."""
    scores = _empty_scores()
    for key, value in raw.items():
        scores[map_emotion(key)] += float(value) * scale
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total
    return scores


class EmotionDetector:
    """Detects the largest face, estimates a mood distribution, and smooths it over time."""

    def __init__(
        self, backend: str = "auto", smoothing_buffer: int = 10, smoothing_window: int = 6
    ) -> None:
        self.smoother = EmotionSmoother(smoothing_buffer, smoothing_window)
        self.last = "neutral"
        self.last_conf = 0.0
        self.active_backend = "none"  # which backend produced the latest result

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        self._fer: Any = None
        self._deepface: Any = None
        if backend in ("auto", "fer"):
            self._fer = self._load_fer()
        if backend in ("auto", "deepface"):
            self._deepface = self._load_deepface()
        if self._fer is None and self._deepface is None and backend != "haar":
            log.warning("No ML emotion backend available; using the Haar smile fallback.")
        log.info(
            "Detector ready (requested=%s, fer=%s, deepface=%s)",
            backend,
            self._fer is not None,
            self._deepface is not None,
        )

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_fer() -> Any:
        try:
            from fer import FER  # optional dependency
        except Exception as exc:  # ImportError, or TF/moviepy errors at import time
            log.info("FER not available: %s", exc)
            return None
        try:
            return FER(mtcnn=True)  # better boxes, slower
        except Exception as exc:
            log.warning("FER(mtcnn=True) failed (%s); retrying with mtcnn=False", exc)
            try:
                return FER(mtcnn=False)
            except Exception as exc2:
                log.warning("FER unavailable: %s", exc2)
                return None

    @staticmethod
    def _load_deepface() -> Any:
        try:
            from deepface import DeepFace  # optional dependency
        except Exception as exc:
            log.info("DeepFace not available: %s", exc)
            return None
        return DeepFace

    # ------------------------------------------------------------------ helpers
    def _find_face(self, gray: np.ndarray) -> Optional[Box]:
        """Largest Haar face, or None. Runs once per frame."""
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        return int(x), int(y), int(w), int(h)

    def _has_smile(self, gray: np.ndarray, face: Box) -> bool:
        """Smile cascade on the lower 45% of ``face`` (coordinates relative to ``gray``)."""
        x, y, w, h = face
        mouth = gray[y + int(h * 0.55) : y + h, x : x + w]
        if mouth.size == 0:
            return False
        try:
            return len(self._smile_cascade.detectMultiScale(mouth, 1.7, 20)) >= 1
        except cv2.error as exc:
            log.debug("smile cascade failed: %s", exc)
            return False

    @staticmethod
    def _padded(face: Box, frame_shape: Tuple[int, ...]) -> Box:
        x, y, w, h = face
        pw, ph = int(w * FACE_PAD), int(h * FACE_PAD)
        x0, y0 = max(0, x - pw), max(0, y - ph)
        x1, y1 = min(frame_shape[1], x + w + pw), min(frame_shape[0], y + h + ph)
        return x0, y0, x1 - x0, y1 - y0

    def _apply_smile_nudge(
        self,
        gray: np.ndarray,
        face: Optional[Box],
        dom: str,
        conf: float,
        scores: Scores,
        threshold: float,
    ) -> Tuple[str, float]:
        """If the model says an unsure 'neutral' but a smile is visible, call it 'happy'."""
        if dom == "neutral" and conf < threshold and face is not None:
            if self._has_smile(gray, face):
                scores["happy"] = max(scores["happy"], SMILE_SCORE)
                return "happy", max(conf, SMILE_SCORE)
        return dom, conf

    # ------------------------------------------------------------------ backends
    def _run_fer(
        self,
        rgb_crop: np.ndarray,
        gray: np.ndarray,
        face: Optional[Box],
        padded: Optional[Box],
        origin: Tuple[int, int],
    ) -> Optional[Detection]:
        try:
            res: List[Dict[str, Any]] = self._fer.detect_emotions(rgb_crop)
        except Exception as exc:
            log.debug("FER inference failed: %s", exc)
            return None
        if not res:
            return None
        emotions = res[0].get("emotions", {})
        if not emotions:
            return None

        scores = _fold_scores(emotions)
        dom_raw = max(emotions, key=emotions.get)
        conf = float(emotions[dom_raw])
        dom = map_emotion(dom_raw)

        box = padded
        fer_box = res[0].get("box")
        if fer_box:
            bx, by, bw, bh = fer_box
            box = (origin[0] + int(bx), origin[1] + int(by), int(bw), int(bh))

        dom, conf = self._apply_smile_nudge(gray, face, dom, conf, scores, SMILE_NUDGE_FER)
        self.active_backend = "fer"
        return Detection(dom, conf, box, scores)

    def _run_deepface(
        self, rgb_crop: np.ndarray, gray: np.ndarray, face: Optional[Box], padded: Optional[Box]
    ) -> Optional[Detection]:
        try:
            analysis = self._deepface.analyze(
                rgb_crop, actions=["emotion"], enforce_detection=False, detector_backend="opencv"
            )
        except Exception as exc:
            log.debug("DeepFace inference failed: %s", exc)
            return None
        # Recent DeepFace versions return a list of dicts, older ones a single dict.
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        emotions = analysis.get("emotion", {})
        if not emotions:
            return None

        scores = _fold_scores(emotions, scale=1 / 100.0)  # DeepFace reports 0..100
        dom_raw = analysis.get("dominant_emotion", "neutral")
        conf = float(emotions.get(dom_raw, 0.0)) / 100.0
        dom = map_emotion(dom_raw)

        dom, conf = self._apply_smile_nudge(gray, face, dom, conf, scores, SMILE_NUDGE_DEEPFACE)
        self.active_backend = "deepface"
        return Detection(dom, conf, padded, scores)

    def _run_haar(self, gray: np.ndarray, face: Optional[Box], padded: Optional[Box]) -> Detection:
        """Last resort: a smile means happy, otherwise repeat the last known mood."""
        if face is None:
            self.active_backend = "none"
            return Detection("neutral", 0.0, None, _empty_scores(neutral=1.0))
        self.active_backend = "haar"
        if self._has_smile(gray, face):
            scores = _empty_scores(neutral=0.2)
            scores["happy"] = 0.8
            return Detection("happy", 0.55, padded, scores)
        return Detection(self.last, 0.45, padded, _empty_scores(neutral=1.0))

    # ------------------------------------------------------------------ public API
    def detect(self, frame: Optional[np.ndarray]) -> Detection:
        """One raw (unsmoothed) detection on a BGR frame."""
        if frame is None:
            return Detection("neutral", 0.0, None, _empty_scores(neutral=1.0))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self._find_face(gray)

        if face is not None:
            padded = self._padded(face, frame.shape)
            px, py, pw, ph = padded
            crop = frame[py : py + ph, px : px + pw]
            origin = (px, py)
        else:
            padded, crop, origin = None, frame, (0, 0)

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self._fer is not None:
            result = self._run_fer(rgb_crop, gray, face, padded, origin)
            if result is not None:
                return result
        if self._deepface is not None:
            result = self._run_deepface(rgb_crop, gray, face, padded)
            if result is not None:
                return result
        return self._run_haar(gray, face, padded)

    def detect_stable(self, frame: Optional[np.ndarray]) -> Detection:
        """Detection smoothed over recent frames: ``(mood, confidence, box, averaged_scores)``."""
        det = self.detect(frame)
        conf = min(1.0, max(0.0, float(det.confidence)))
        stable, stable_conf, scores = self.smoother.update(det.scores, det.emotion, conf)
        self.last = stable
        self.last_conf = stable_conf
        return Detection(stable, stable_conf, det.box, scores)
