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
    out = det.detect_stable(blank)
    assert out.box is None
    assert out.emotion == "neutral"
    assert 0.0 <= out.confidence <= 1.0
    assert sum(out.scores.values()) > 0
    assert det.active_backend == "none"


def test_detect_none_frame_is_safe():
    det = EmotionDetector(backend="haar")
    assert det.detect(None).emotion == "neutral"


# ---------------------------------------------------------------- MediaPipe path (fake backend)
from narrative_engine.perception.mediapipe_backend import FaceObservation  # noqa: E402

BOX = (40, 30, 100, 100)
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


class FakeMP:
    def __init__(self):
        self.blend = {}
        self.visible = True
        self.closed = False

    def observe(self, rgb):
        return FaceObservation(BOX, dict(self.blend)) if self.visible else None

    def close(self):
        self.closed = True


def make_detector():
    det = EmotionDetector(backend="haar")
    det._mp = FakeMP()
    return det


def test_mediapipe_smile_is_happy_with_positive_valence():
    det = make_detector()
    det._mp.blend = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
    result = None
    for _ in range(12):
        result = det.detect_stable(FRAME)
    assert result.emotion == "happy" and result.box == BOX
    assert result.valence > 0.3 and det.active_backend == "mediapipe"


def test_mediapipe_no_face_gives_no_box_and_affect_decays():
    det = make_detector()
    det._mp.blend = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
    for _ in range(12):
        det.detect_stable(FRAME)
    det._mp.visible = False
    out = [det.detect_stable(FRAME) for _ in range(30)]
    assert out[-1].box is None
    assert abs(out[-1].valence) < 0.01  # decayed toward neutral


def test_calibration_removes_a_resting_smile():
    det = make_detector()
    assert det.supports_calibration and not det.calibrated
    det._mp.blend = {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5}  # resting face looks smiley
    det.begin_calibration()
    for _ in range(15):
        det.detect(FRAME)
    assert det.finish_calibration() and det.calibrated
    result = None
    for _ in range(12):
        result = det.detect_stable(FRAME)
    assert result.emotion == "neutral" and abs(result.valence) < 0.05


def test_calibration_fails_with_too_few_samples_and_keeps_old_baseline():
    det = make_detector()
    det._mp.visible = False
    det.begin_calibration()
    for _ in range(15):
        det.detect(FRAME)
    assert det.finish_calibration() is False and not det.calibrated


def test_haar_only_detector_cannot_calibrate_and_close_is_safe():
    det = EmotionDetector(backend="haar")
    assert not det.supports_calibration
    det.close()
    fake = make_detector()
    fake.close()
    assert fake._mp.closed


def test_detection_carries_valence_and_arousal_fields():
    d = EmotionDetector(backend="haar").detect(None)
    assert d.valence == 0.0 and d.arousal == 0.0
