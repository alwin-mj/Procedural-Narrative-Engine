from types import SimpleNamespace

import numpy as np
import pytest

from narrative_engine.perception import mediapipe_backend as mb


def test_explicit_missing_model_path_raises(tmp_path):
    with pytest.raises(mb.ModelUnavailable):
        mb.ensure_model(tmp_path / "nope.task")


def test_explicit_existing_model_path_is_returned(tmp_path):
    f = tmp_path / "m.task"
    f.write_bytes(b"x")
    assert mb.ensure_model(f) == f


def test_failed_download_raises_and_leaves_no_partial_file(tmp_path):
    # unreachable URL scheme -> URLError/OSError -> ModelUnavailable, no network needed
    mb_default = mb.default_model_path
    mb.default_model_path = lambda: tmp_path / "cache" / "face_landmarker.task"
    try:
        with pytest.raises(mb.ModelUnavailable):
            mb.ensure_model(url="file:///definitely/not/here.task", timeout=1.0)
    finally:
        mb.default_model_path = mb_default
    assert not list((tmp_path / "cache").glob("*.part"))


def _backend_with(result):
    backend = mb.MediaPipeBackend.__new__(mb.MediaPipeBackend)  # skip real model loading
    backend._mp = SimpleNamespace(
        Image=lambda image_format, data: SimpleNamespace(data=data),
        ImageFormat=SimpleNamespace(SRGB=1),
    )
    calls = []

    def detect_for_video(image, ts):
        calls.append(ts)
        return result

    backend._landmarker = SimpleNamespace(detect_for_video=detect_for_video, close=lambda: None)
    backend._last_ts_ms = 0
    return backend, calls


def test_observe_returns_box_and_blendshapes():
    pts = [SimpleNamespace(x=0.25, y=0.25), SimpleNamespace(x=0.75, y=0.5)]
    cats = [SimpleNamespace(category_name="mouthSmileLeft", score=0.6)]
    backend, _ = _backend_with(SimpleNamespace(face_landmarks=[pts], face_blendshapes=[cats]))
    obs = backend.observe(np.zeros((100, 200, 3), dtype=np.uint8))
    assert obs.box == (50, 25, 100, 25)
    assert obs.blendshapes == {"mouthSmileLeft": pytest.approx(0.6)}


def test_observe_returns_none_without_a_face_and_timestamps_increase():
    backend, calls = _backend_with(SimpleNamespace(face_landmarks=[], face_blendshapes=[]))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert backend.observe(frame) is None
    for _ in range(5):
        backend.observe(frame)
    assert all(b > a for a, b in zip(calls, calls[1:]))


def _with_cache(tmp_path):
    orig = mb.default_model_path
    mb.default_model_path = lambda: tmp_path / "cache" / "face_landmarker.task"
    return orig


def test_download_success_caches_the_file_and_is_reused(tmp_path):
    src = tmp_path / "remote.task"
    src.write_bytes(b"x" * (mb.MIN_MODEL_BYTES + 10))
    orig = _with_cache(tmp_path)
    try:
        path = mb.ensure_model(url=src.as_uri())
        assert path.is_file() and path.stat().st_size == src.stat().st_size
        assert mb.ensure_model(url="file:///definitely/not/here") == path  # served from cache
        assert not list(path.parent.glob("*.part"))
    finally:
        mb.default_model_path = orig


def test_truncated_download_is_rejected_and_cleaned_up(tmp_path):
    src = tmp_path / "tiny.task"
    src.write_bytes(b"x" * 100)
    orig = _with_cache(tmp_path)
    try:
        with pytest.raises(mb.ModelUnavailable):
            mb.ensure_model(url=src.as_uri())
    finally:
        mb.default_model_path = orig
    assert not list((tmp_path / "cache").glob("*"))


def test_close_releases_the_landmarker():
    closed = []
    backend, _ = _backend_with(SimpleNamespace(face_landmarks=[], face_blendshapes=[]))
    backend._landmarker.close = lambda: closed.append(True)
    backend.close()
    assert closed == [True]
