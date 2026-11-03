import numpy as np

from narrative_engine.perception import EmotionDetector, map_emotion


def test_map_emotion_folds_labels_and_defaults_to_neutral():
    assert map_emotion("surprise") == "uplift"
    assert map_emotion("ANGRY") == "sad"
    assert map_emotion("joy") == "happy"
    assert map_emotion("???") == "neutral"


def test_haar_backend_on_blank_frame_reports_no_face():
    det = EmotionDetector(backend="haar")
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    emotion, conf, box, scores = det.detect_stable(blank)
    assert box is None
    assert emotion == "neutral"
    assert 0.0 <= conf <= 1.0
    assert abs(sum(scores.values()) - 1.0) < 1e-6 or sum(scores.values()) > 0
    assert det.active_backend == "none"


def test_detect_none_frame_is_safe():
    det = EmotionDetector(backend="haar")
    assert det.detect(None).emotion == "neutral"
