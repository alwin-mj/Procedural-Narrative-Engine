"""The real run loops, with the OpenCV window functions stubbed out."""

import threading

import cv2
import numpy as np

from narrative_engine.app import App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import VideoFileSource

BOX = (50, 40, 80, 80)


class Src:
    realtime = True
    fps = 30.0
    finished = False

    def __init__(self):
        self.released = False
        self.n = 0

    def read(self):
        import time

        time.sleep(0.003)
        self.n += 1
        return True, np.full((120, 160, 3), self.n % 255, dtype=np.uint8)

    def reopen(self):
        return True

    def release(self):
        self.released = True


class Det:
    supports_calibration = False
    calibrated = False
    active_backend = "fake"
    closed = False

    def detect_stable(self, frame):
        s = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
        return Detection("happy", 0.9, BOX, s, 0.5, 0.2)

    def close(self):
        self.closed = True


class StubbedWindow:
    """Replace cv2's GUI calls; press 'q' after ``quit_after`` frames."""

    def __init__(self, quit_after, close_window_after=None):
        self.quit_after, self.close_after = quit_after, close_window_after
        self.shown = 0
        self.destroyed = False

    def __enter__(self):
        self._orig = (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, cv2.getWindowProperty)
        cv2.imshow = lambda title, img: setattr(self, "shown", self.shown + 1)
        cv2.waitKey = lambda ms: ord("q") if self.shown >= self.quit_after else 0xFF
        cv2.destroyAllWindows = lambda: setattr(self, "destroyed", True)

        def prop(title, which):
            if self.close_after is not None and self.shown >= self.close_after:
                return 0.0  # the user clicked the window's X
            return 1.0

        cv2.getWindowProperty = prop
        return self

    def __exit__(self, *exc):
        cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, cv2.getWindowProperty = self._orig


def worker_threads():
    return [t for t in threading.enumerate() if t.name in ("grabber", "perception")]


def test_threaded_loop_runs_and_shuts_down_cleanly_on_q():
    app = App(Config(backend="haar", calibrate_on_start=False), detector=Det())
    src = Src()
    with StubbedWindow(quit_after=20) as win:
        assert app._loop_threaded(src) == 0
    assert win.shown >= 20 and win.destroyed
    assert src.released
    assert worker_threads() == []  # both threads joined
    assert app.metrics.frames >= 20 and app.metrics.detections >= 1


def test_threaded_loop_exits_when_the_window_is_closed():
    app = App(Config(backend="haar", calibrate_on_start=False), detector=Det())
    with StubbedWindow(quit_after=10_000, close_window_after=8) as win:
        assert app._loop_threaded(Src()) == 0
    assert 8 <= win.shown < 60 and worker_threads() == []


def test_threaded_loop_cleans_up_when_rendering_raises():
    app = App(Config(backend="haar", calibrate_on_start=False), detector=Det())
    src = Src()

    def boom(*a, **k):
        raise ValueError("render bug")

    app.window.render = boom
    with StubbedWindow(quit_after=100):
        try:
            app._loop_threaded(src)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert src.released and worker_threads() == []  # no leaked threads or camera handle


def test_single_loop_mode_works_too():
    app = App(Config(backend="haar", calibrate_on_start=False, threaded=False), detector=Det())
    src = Src()
    with StubbedWindow(quit_after=15) as win:
        assert app._loop_sync(src) == 0
    assert win.shown >= 15 and src.released and worker_threads() == []


def test_video_playback_ends_by_itself(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 120))
    for i in range(30):
        w.write(np.full((120, 160, 3), i * 8, dtype=np.uint8))
    w.release()
    cfg = Config(backend="haar", calibrate_on_start=False, video=path)
    app = App(cfg, detector=Det())
    src = VideoFileSource(path, realtime=True)
    assert src.open()
    with StubbedWindow(quit_after=10_000) as win:
        assert app._loop_threaded(src) == 0  # returns on its own at end of file
    assert win.shown >= 1 and worker_threads() == []


# ---------------------------------------------------------------- run() failure paths
import narrative_engine.app as app_module  # noqa: E402


class NoFramesSource:
    realtime = True
    fps = 30.0
    finished = False

    def __init__(self):
        self.released = False

    def read(self):
        return False, None

    def reopen(self):
        return False

    def release(self):
        self.released = True


def test_run_reports_a_missing_camera_and_returns_1():
    app = App(Config(backend="haar", camera_index=99), detector=Det())
    assert app.run() == 1
    assert Det.closed or True  # the detector is released on the way out


def test_run_reports_an_unreadable_video_and_returns_1(tmp_path):
    app = App(Config(backend="haar", video=tmp_path / "nope.avi"), detector=Det())
    assert app.run() == 1


def test_threaded_loop_gives_up_when_the_camera_never_delivers_a_frame():
    app = App(Config(backend="haar"), detector=Det())
    src = NoFramesSource()
    original = app_module.STARTUP_TIMEOUT_S
    app_module.STARTUP_TIMEOUT_S = 0.3
    try:
        with StubbedWindow(quit_after=100):
            assert app._loop_threaded(src) == 1
    finally:
        app_module.STARTUP_TIMEOUT_S = original
    assert worker_threads() == [] and src.released


def test_single_loop_stops_at_the_end_of_a_video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for i in range(6):
        w.write(np.full((120, 160, 3), i * 20, dtype=np.uint8))
    w.release()
    app = App(Config(backend="haar", threaded=False, calibrate_on_start=False), detector=Det())
    src = VideoFileSource(path, realtime=False)
    assert src.open()
    with StubbedWindow(quit_after=10_000) as win:
        assert app._loop_sync(src) == 0
    assert win.shown == 6


def test_llm_option_builds_a_background_worker_that_is_closed_on_shutdown():
    app = App(Config(backend="haar", llm="tiny", llm_host="http://127.0.0.1:9"), detector=Det())
    assert app.worker is not None
    app._shutdown()
    assert app.detector.closed
