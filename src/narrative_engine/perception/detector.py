"""Face + mood detection with an optional-backend fallback chain.

Chain (``backend="auto"``): MediaPipe -> FER -> DeepFace -> Haar face + smile heuristic.
Every ML backend is optional; if one is missing or fails, the next takes over, so the app
always starts. Each result carries the four story-mood scores *and* continuous
valence/arousal.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from ..config import MOODS
from .affect import (
    baseline_from_samples,
    circumplex_va,
    estimate_affect,
    extract_features,
    scores_from_affect,
)
from .smoothing import Ema, EmotionSmoother, HysteresisGate, Scores

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x, y, w, h

# Raw label -> one of the four story moods. The story has four mood variants per scene, so the
# classic 7 emotions are folded down. angry / fear / disgust all land in "sad" (the low-valence
# bucket); their distinct valence/arousal is still reported through ``Detection.valence/arousal``.
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
SMILE_NUDGE_FER = 0.70  # FER: nudge unsure neutral -> happy when confidence is below this
SMILE_NUDGE_DEEPFACE = 0.65
SMILE_SCORE = 0.55  # happy score assigned when the smile cue fires
MIN_CALIBRATION_SAMPLES = 10


class Detection(NamedTuple):
    """Result of one detection step."""

    emotion: str
    confidence: float
    box: Optional[Box]
    scores: Scores
    valence: float = 0.0  # -1 unhappy .. +1 happy
    arousal: float = 0.0  # calm .. excited


def map_emotion(raw: Any) -> str:
    """Fold a raw model label into one of the four story moods (unknown -> neutral)."""
    return EMOTION_MAP.get(str(raw).lower(), "neutral")


def _empty_scores(neutral: float = 0.0) -> Scores:
    s: Scores = {m: 0.0 for m in MOODS}
    s["neutral"] = neutral
    return s


def _no_face() -> Detection:
    return Detection("neutral", 0.0, None, _empty_scores(neutral=1.0), 0.0, 0.0)


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
    """Detects the largest face, estimates mood + valence/arousal, and smooths them over time."""

    def __init__(
        self,
        backend: str = "auto",
        smoothing_buffer: int = 10,
        smoothing_window: int = 6,
        *,
        hysteresis_margin: float = 0.08,
        hysteresis_frames: int = 3,
        affect_alpha: float = 0.3,
        mediapipe_model: Optional[Path] = None,
    ) -> None:
        self.smoother = EmotionSmoother(smoothing_buffer, smoothing_window)
        self.gate = HysteresisGate(hysteresis_margin, hysteresis_frames)
        self._valence = Ema(affect_alpha)
        self._arousal = Ema(affect_alpha)
        self.last = "neutral"
        self.last_conf = 0.0
        self.active_backend = "none"  # which backend produced the latest result
        # Per-stage timings (ms) of the most recent frame, for benchmarking:
        # face_ms (Haar face search), infer_ms (backend inference), smooth_ms (smoothing).
        self.last_timings: Dict[str, float] = {"face_ms": 0.0, "infer_ms": 0.0, "smooth_ms": 0.0}

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        self._mp: Any = None
        self._fer: Any = None
        self._deepface: Any = None
        self._baseline: Optional[Dict[str, float]] = None
        self._calib_samples: Optional[List[Dict[str, float]]] = None
        # Detection may run on a worker thread while the UI thread starts/finishes calibration
        # or resets smoothing; this guards that shared state (inference itself is not locked).
        self._state_lock = threading.RLock()

        if backend in ("auto", "mediapipe"):
            self._mp = self._load_mediapipe(mediapipe_model)
        # Only load the heavy TensorFlow-based backends when MediaPipe is not doing the job.
        want_ml = backend in ("fer", "deepface") or (backend == "auto" and self._mp is None)
        if want_ml and backend in ("auto", "fer"):
            self._fer = self._load_fer()
        if want_ml and backend in ("auto", "deepface"):
            self._deepface = self._load_deepface()

        if backend != "haar" and not (self._mp or self._fer or self._deepface):
            log.warning("No ML emotion backend available; using the Haar smile fallback.")
        log.info(
            "Detector ready (requested=%s, mediapipe=%s, fer=%s, deepface=%s)",
            backend,
            self._mp is not None,
            self._fer is not None,
            self._deepface is not None,
        )

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_mediapipe(model_path: Optional[Path]) -> Any:
        try:
            from .mediapipe_backend import MediaPipeBackend

            return MediaPipeBackend(model_path)
        except ImportError as exc:
            log.info("MediaPipe not installed (%s). Try: pip install -e '.[mediapipe]'", exc)
        except Exception as exc:  # ModelUnavailable, or a MediaPipe runtime error
            log.warning("MediaPipe backend unavailable: %s", exc)
        return None

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

    def close(self) -> None:
        """Release backend resources."""
        if self._mp is not None:
            try:
                self._mp.close()
            except Exception as exc:
                log.debug("MediaPipe close failed: %s", exc)

    # ------------------------------------------------------------------ calibration
    @property
    def loaded_backends(self) -> List[str]:
        """Names of the backends that actually loaded (``haar`` is always available)."""
        names = [n for n, obj in (("mediapipe", self._mp), ("fer", self._fer),
                                  ("deepface", self._deepface)) if obj is not None]
        return names + ["haar"]

    @property
    def supports_calibration(self) -> bool:
        """Calibration needs blendshape features, which only the MediaPipe backend provides."""
        return self._mp is not None

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    def begin_calibration(self) -> None:
        """Start collecting the user's resting (neutral) face."""
        with self._state_lock:
            self._calib_samples = []

    def finish_calibration(self) -> bool:
        """Store the baseline. Returns False (keeping the old baseline) if too few samples."""
        with self._state_lock:
            samples, self._calib_samples = self._calib_samples or [], None
        if len(samples) < MIN_CALIBRATION_SAMPLES:
            log.warning("Calibration failed: only %d face samples", len(samples))
            return False
        self._baseline = baseline_from_samples(samples)
        self.reset_smoothing()
        log.info("Calibrated on %d samples", len(samples))
        return True

    def reset_smoothing(self) -> None:
        with self._state_lock:
            self.smoother.reset()
            self.gate.reset()
            self._valence.reset()
            self._arousal.reset()

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
        """FER/DeepFace only: an unsure 'neutral' plus a visible smile becomes 'happy'."""
        if dom == "neutral" and conf < threshold and face is not None:
            if self._has_smile(gray, face):
                scores["happy"] = max(scores["happy"], SMILE_SCORE)
                return "happy", max(conf, SMILE_SCORE)
        return dom, conf

    # ------------------------------------------------------------------ backends
    def _run_mediapipe(self, frame: np.ndarray) -> Optional[Detection]:
        """Blendshapes -> features -> valence/arousal -> mood scores. None means 'try next'."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            obs = self._mp.observe(rgb)
        except Exception as exc:
            log.debug("MediaPipe inference failed: %s", exc)
            return None
        self.active_backend = "mediapipe"
        if obs is None:
            return _no_face()

        features = extract_features(obs.blendshapes)
        with self._state_lock:
            if self._calib_samples is not None:
                self._calib_samples.append(features)
        valence, arousal = estimate_affect(features, self._baseline)
        scores = scores_from_affect(valence, arousal)
        dom = max(scores, key=scores.get)  # type: ignore[arg-type]
        return Detection(dom, scores[dom], obs.box, scores, valence, arousal)

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
        valence, arousal = circumplex_va(emotions)
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
        return Detection(dom, conf, box, scores, valence, arousal)

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
        valence, arousal = circumplex_va(emotions)
        dom_raw = analysis.get("dominant_emotion", "neutral")
        conf = float(emotions.get(dom_raw, 0.0)) / 100.0
        dom = map_emotion(dom_raw)

        dom, conf = self._apply_smile_nudge(gray, face, dom, conf, scores, SMILE_NUDGE_DEEPFACE)
        self.active_backend = "deepface"
        return Detection(dom, conf, padded, scores, valence, arousal)

    def _run_haar(self, gray: np.ndarray, face: Optional[Box], padded: Optional[Box]) -> Detection:
        """Last resort: a smile means happy, otherwise repeat the last known mood."""
        if face is None:
            self.active_backend = "none"
            return _no_face()
        self.active_backend = "haar"
        if self._has_smile(gray, face):
            scores = _empty_scores(neutral=0.2)
            scores["happy"] = 0.8
            valence, arousal = circumplex_va({"happy": 1.0})
            return Detection("happy", 0.55, padded, scores, valence, arousal)
        return Detection(self.last, 0.45, padded, _empty_scores(neutral=1.0), 0.0, 0.0)

    # ------------------------------------------------------------------ public API
    def detect(self, frame: Optional[np.ndarray]) -> Detection:
        """One raw (unsmoothed) detection on a BGR frame."""
        if frame is None:
            return _no_face()

        t0 = time.perf_counter()
        if self._mp is not None:
            result = self._run_mediapipe(frame)
            if result is not None:
                # MediaPipe finds the face and reads expressions in one model: all "infer".
                self.last_timings.update(face_ms=0.0, infer_ms=(time.perf_counter() - t0) * 1e3)
                return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self._find_face(gray)
        t_face = time.perf_counter()
        self.last_timings["face_ms"] = (t_face - t0) * 1e3

        if face is not None:
            padded = self._padded(face, frame.shape)
            px, py, pw, ph = padded
            crop = frame[py : py + ph, px : px + pw]
            origin = (px, py)
        else:
            padded, crop, origin = None, frame, (0, 0)

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = None
        if self._fer is not None:
            result = self._run_fer(rgb_crop, gray, face, padded, origin)
        if result is None and self._deepface is not None:
            result = self._run_deepface(rgb_crop, gray, face, padded)
        if result is None:
            result = self._run_haar(gray, face, padded)
        self.last_timings["infer_ms"] = (time.perf_counter() - t_face) * 1e3
        return result

    def detect_stable(self, frame: Optional[np.ndarray]) -> Detection:
        """Detection smoothed over time (rolling average, hysteresis, EMA valence/arousal)."""
        return self.stabilize(self.detect(frame))

    def stabilize(self, det: Detection) -> Detection:
        """Apply temporal smoothing to a raw detection.

        Split out of :meth:`detect_stable` so an evaluation can get the raw and the smoothed
        result from a single model inference.
        """
        t0 = time.perf_counter()
        conf = min(1.0, max(0.0, float(det.confidence)))
        with self._state_lock:
            candidate, cand_conf, avg = self.smoother.update(det.scores, det.emotion, conf)
            mood = self.gate.update(candidate, avg)
            mood_conf = cand_conf if mood == candidate else avg.get(mood, cand_conf)

            # With no face, valence/arousal decay toward neutral instead of freezing.
            has_face = det.box is not None
            valence = self._valence.update(det.valence if has_face else 0.0)
            arousal = self._arousal.update(det.arousal if has_face else 0.0)

            self.last = mood
            self.last_conf = mood_conf
        self.last_timings["smooth_ms"] = (time.perf_counter() - t0) * 1e3
        return Detection(mood, mood_conf, det.box, avg, valence, arousal)
