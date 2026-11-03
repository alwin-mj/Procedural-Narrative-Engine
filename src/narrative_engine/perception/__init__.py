"""Perception: turn camera frames into a stable mood + valence/arousal estimate."""

from .affect import (
    estimate_affect,
    extract_features,
    scores_from_affect,
)
from .detector import Detection, EmotionDetector, map_emotion
from .motion import FaceMotion, MotionDetector
from .smoothing import Ema, EmotionSmoother, HysteresisGate

__all__ = [
    "Detection",
    "Ema",
    "EmotionDetector",
    "EmotionSmoother",
    "FaceMotion",
    "HysteresisGate",
    "MotionDetector",
    "estimate_affect",
    "extract_features",
    "map_emotion",
    "scores_from_affect",
]
