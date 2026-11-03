"""Perception: turn camera frames into a stable mood estimate."""

from .detector import Detection, EmotionDetector, map_emotion
from .motion import MotionDetector
from .smoothing import EmotionSmoother

__all__ = ["Detection", "EmotionDetector", "EmotionSmoother", "MotionDetector", "map_emotion"]
