"""The FER / DeepFace / Haar paths and loaders, exercised with fake models (no TensorFlow)."""

import sys
import types

import cv2
import numpy as np

from narrative_engine.perception import EmotionDetector
from narrative_engine.perception.detector import _fold_scores

FRAME = np.full((240, 320, 3), 90, dtype=np.uint8)
FER_EMOTIONS = {"angry": 0.10, "disgust": 0.05, "fear": 0.05, "happy": 0.55, "sad": 0.05,
                "surprise": 0.05, "neutral": 0.15}


class FakeFER:
    def __init__(self, result=None, raises=False):
        self.result, self.raises, self.calls = result, raises, 0

    def detect_emotions(self, rgb):
        self.calls += 1
        if self.raises:
            raise RuntimeError("fer broke")
        return self.result


class FakeDeepFace:
    def __init__(self, result=None, raises=False):
        self.result, self.raises = result, raises

    def analyze(self, img, actions, enforce_detection, detector_backend):
        assert actions == ["emotion"] and enforce_detection is False
        if self.raises:
            raise ValueError("deepface broke")
        return self.result


def fer_result(emotions=None, box=(10, 10, 80, 80)):
    return [{"box": box, "emotions": dict(emotions or FER_EMOTIONS)}]


def deepface_result(dominant="happy", pct=None):
    emotions = pct or {"angry": 2.0, "disgust": 1.0, "fear": 2.0, "happy": 80.0, "sad": 5.0,
                       "surprise": 5.0, "neutral": 5.0}
    return {"emotion": emotions, "dominant_emotion": dominant}


def detector(**fakes):
    det = EmotionDetector(backend="haar")
    det._fer = fakes.get("fer")
    det._deepface = fakes.get("deepface")
    det._find_face = fakes.get("face", lambda gray: None)
    return det


# ---------------------------------------------------------------- score folding
def test_fold_scores_sums_into_four_moods_and_normalises():
    s = _fold_scores(FER_EMOTIONS)
    assert abs(sum(s.values()) - 1.0) < 1e-9
    assert abs(s["sad"] - 0.25) < 1e-9  # angry + disgust + fear + sad all land in 'sad'
    assert abs(s["uplift"] - 0.05) < 1e-9 and abs(s["happy"] - 0.55) < 1e-9
    assert _fold_scores({}) == {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}


# ---------------------------------------------------------------- FER
def test_fer_result_becomes_a_detection_with_valence_and_arousal():
    det = detector(fer=FakeFER(fer_result()))
    out = det.detect(FRAME)
    assert out.emotion == "happy" and abs(out.confidence - 0.55) < 1e-9
    assert out.box == (10, 10, 80, 80)  # no Haar face: crop is the full frame, no offset
    assert out.valence > 0 and det.active_backend == "fer"


def test_fer_box_is_offset_by_the_padded_face_crop():
    face = lambda gray: (60, 50, 100, 100)  # noqa: E731
    det = detector(fer=FakeFER(fer_result(box=(10, 10, 80, 80))), face=face)
    out = det.detect(FRAME)
    # face (60,50,100,100) padded by 12% -> crop origin (48,38); FER box is relative to the crop
    assert out.box == (58, 48, 80, 80)


def test_fer_without_its_own_box_falls_back_to_the_padded_haar_box():
    no_box = FakeFER([{"emotions": dict(FER_EMOTIONS)}])
    det = detector(fer=no_box, face=lambda gray: (60, 50, 100, 100))
    assert det.detect(FRAME).box == (48, 38, 124, 124)


def test_unsure_neutral_plus_smile_becomes_happy_but_only_below_the_threshold():
    neutralish = {"neutral": 0.5, "sad": 0.2, "happy": 0.2, "angry": 0.1}
    det = detector(fer=FakeFER(fer_result(neutralish)), face=lambda gray: (60, 50, 100, 100))
    det._has_smile = lambda gray, face: True
    out = det.detect(FRAME)
    assert out.emotion == "happy" and out.confidence >= 0.55 and out.scores["happy"] >= 0.55
    det._has_smile = lambda gray, face: False
    assert det.detect(FRAME).emotion == "neutral"
    sure = {"neutral": 0.9, "sad": 0.1}
    det2 = detector(fer=FakeFER(fer_result(sure)), face=lambda gray: (60, 50, 100, 100))
    det2._has_smile = lambda gray, face: True
    assert det2.detect(FRAME).emotion == "neutral"  # 0.9 >= 0.70: trust the model


def test_fer_empty_or_failing_falls_through_to_the_next_backend():
    deep = FakeDeepFace(deepface_result())
    for fer in (FakeFER([]), FakeFER([{"emotions": {}}]), FakeFER(raises=True)):
        det = detector(fer=fer, deepface=deep)
        assert det.detect(FRAME).emotion == "happy" and det.active_backend == "deepface"


# ---------------------------------------------------------------- DeepFace
def test_deepface_list_output_is_handled():
    """Recent DeepFace versions return a list of dicts; this used to be silently swallowed."""
    det = detector(deepface=FakeDeepFace([deepface_result()]))
    out = det.detect(FRAME)
    assert out.emotion == "happy" and abs(out.confidence - 0.80) < 1e-9
    assert abs(sum(out.scores.values()) - 1.0) < 1e-9 and det.active_backend == "deepface"


def test_deepface_dict_output_still_works_and_uses_the_padded_box():
    det = detector(deepface=FakeDeepFace(deepface_result("sad", {"sad": 70.0, "neutral": 30.0})),
                   face=lambda gray: (60, 50, 100, 100))
    out = det.detect(FRAME)
    assert out.emotion == "sad" and out.valence < 0 and out.box == (48, 38, 124, 124)


def test_deepface_empty_list_or_failure_falls_back_to_haar():
    for df in (FakeDeepFace([]), FakeDeepFace({}), FakeDeepFace(raises=True)):
        det = detector(deepface=df)
        out = det.detect(FRAME)
        assert out.box is None and det.active_backend == "none"


# ---------------------------------------------------------------- Haar fallback
def test_haar_smile_is_happy_otherwise_the_last_mood_is_repeated():
    det = detector(face=lambda gray: (60, 50, 100, 100))
    det._has_smile = lambda gray, face: True
    out = det.detect(FRAME)
    assert out.emotion == "happy" and out.confidence == 0.55 and out.valence > 0
    det._has_smile = lambda gray, face: False
    det.last = "sad"
    out = det.detect(FRAME)
    assert out.emotion == "sad" and out.confidence == 0.45 and det.active_backend == "haar"


def test_find_face_picks_the_largest_candidate():
    det = EmotionDetector(backend="haar")
    gray = np.zeros((100, 100), dtype=np.uint8)
    det._face_cascade = types.SimpleNamespace(
        detectMultiScale=lambda g, a, b: np.array([[0, 0, 10, 10], [5, 5, 40, 30], [1, 1, 20, 20]])
    )
    assert det._find_face(gray) == (5, 5, 40, 30)
    det._face_cascade = types.SimpleNamespace(detectMultiScale=lambda g, a, b: ())
    assert det._find_face(gray) is None


def test_has_smile_handles_empty_regions_and_opencv_errors():
    det = EmotionDetector(backend="haar")
    gray = np.zeros((50, 50), dtype=np.uint8)
    assert det._has_smile(gray, (10, 10, 20, 0)) is False  # zero-height face -> empty mouth region

    def boom(*a):
        raise cv2.error("bad input")

    det._smile_cascade = types.SimpleNamespace(detectMultiScale=boom)
    assert det._has_smile(gray, (10, 10, 20, 20)) is False
    det._smile_cascade = types.SimpleNamespace(detectMultiScale=lambda *a: [[1, 1, 5, 5]])
    assert det._has_smile(gray, (10, 10, 20, 20)) is True


def test_padded_box_is_clamped_to_the_frame():
    assert EmotionDetector._padded((0, 0, 100, 100), (240, 320, 3)) == (0, 0, 112, 112)
    assert EmotionDetector._padded((250, 150, 100, 100), (240, 320, 3)) == (238, 138, 82, 102)


# ---------------------------------------------------------------- loaders
class BlockedImport:
    """Make ``import <name>`` raise ImportError for the duration of a with-block."""

    def __init__(self, *names):
        self.names, self.saved = names, {}

    def __enter__(self):
        for n in self.names:
            self.saved[n] = sys.modules.get(n, "absent")
            sys.modules[n] = None
        return self

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            if v == "absent":
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


class FakeModule:
    def __init__(self, **attrs):
        self.attrs = attrs

    def __enter__(self):
        self.saved = {n: sys.modules.get(n, "absent") for n in self.attrs}
        for n, mod in self.attrs.items():
            sys.modules[n] = mod
        return self

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            if v == "absent":
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def test_missing_optional_backends_load_as_none():
    with BlockedImport("fer", "deepface", "mediapipe"):
        assert EmotionDetector._load_fer() is None
        assert EmotionDetector._load_deepface() is None
        assert EmotionDetector._load_mediapipe(None) is None  # ImportError path
        det = EmotionDetector(backend="auto")
        assert det.active_backend == "none" and not det.supports_calibration


def test_mediapipe_loader_survives_a_missing_model_file(tmp_path):
    assert EmotionDetector._load_mediapipe(tmp_path / "missing.task") is None


def test_fer_loader_retries_without_mtcnn_then_gives_up():
    calls = []

    def flaky(mtcnn):
        calls.append(mtcnn)
        if mtcnn:
            raise RuntimeError("no mtcnn")
        return "fer-without-mtcnn"

    with FakeModule(fer=types.SimpleNamespace(FER=flaky)):
        assert EmotionDetector._load_fer() == "fer-without-mtcnn" and calls == [True, False]

    def always_fails(mtcnn):
        raise RuntimeError("nope")

    with FakeModule(fer=types.SimpleNamespace(FER=always_fails)):
        assert EmotionDetector._load_fer() is None


def test_deepface_loader_returns_the_class():
    marker = object()
    with FakeModule(deepface=types.SimpleNamespace(DeepFace=marker)):
        assert EmotionDetector._load_deepface() is marker


def test_explicit_backend_choices_load_only_what_was_asked_for():
    with FakeModule(fer=types.SimpleNamespace(FER=lambda mtcnn: FakeFER()),
                    deepface=types.SimpleNamespace(DeepFace=FakeDeepFace())):
        assert EmotionDetector(backend="fer")._deepface is None
        assert EmotionDetector(backend="deepface")._fer is None
        both = EmotionDetector(backend="auto")  # MediaPipe model is unavailable offline
        assert both._fer is not None or both._mp is not None


def test_close_swallows_backend_errors():
    class Bad:
        def close(self):
            raise RuntimeError("already closed")

    det = EmotionDetector(backend="haar")
    det._mp = Bad()
    det.close()  # must not raise
