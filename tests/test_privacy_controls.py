"""Camera off/on, the always-visible camera indicator, and demo recording."""

import json
import threading

import cv2
import numpy as np
import pytest

import narrative_engine.app as app_module
from narrative_engine.app import PAUSED_TEXT, STARTING_TEXT, App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import Pipeline, VideoFileSource
from narrative_engine.render import GameWindow

BOX = (50, 40, 80, 80)
FRAME = np.zeros((120, 160, 3), dtype=np.uint8)
SCORES = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}


class Det:
    supports_calibration = False
    calibrated = False
    active_backend = "fake"

    def __init__(self):
        self.calls = self.resets = 0

    def detect_stable(self, frame):
        self.calls += 1
        return Detection("happy", 0.9, BOX, dict(SCORES), 0.5, 0.2)

    def reset_smoothing(self):
        self.resets += 1

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def build(**cfg):
    det, clock = Det(), Clock()
    app = App(Config(backend="haar", calibrate_on_start=False, **cfg), detector=det, clock=clock)
    seen = []
    blank = np.zeros((app.cfg.height, app.cfg.width, 3), dtype=np.uint8)

    def record(frame, emo, conf, box, active, text, *a, **k):
        seen.append({"frame": frame, "text": text, "box": box, **k})
        return blank

    app.window.render = record
    return app, det, clock, seen


def tick(app, clock, seconds=0.1):
    clock.t += seconds
    return app.step(FRAME)


# ---------------------------------------------------------------- the X key
def test_x_switches_the_camera_off_and_nothing_is_processed_or_shown():
    app, det, clock, seen = build()
    tick(app, clock)
    assert det.calls == 1 and seen[-1]["frame"] is not None
    app.handle_key(ord("x"))
    assert app.camera_paused
    for _ in range(5):
        tick(app, clock)
    assert det.calls == 1  # the detector never saw another frame
    assert seen[-1]["frame"] is None and seen[-1]["text"] == PAUSED_TEXT
    assert seen[-1]["camera_on"] is False and seen[-1]["box"] is None


def test_x_again_switches_it_back_on_with_a_clean_slate():
    app, det, clock, seen = build()
    app.handle_key(ord("x"))
    tick(app, clock)
    app.handle_key(ord("x"))
    assert not app.camera_paused and det.resets == 1  # no mood carried over from before
    tick(app, clock)
    assert det.calls == 1 and seen[-1]["camera_on"] is True and seen[-1]["frame"] is not None


def test_the_story_does_not_advance_while_the_camera_is_off_nor_jump_when_it_returns():
    app, det, clock, seen = build(advance_interval_s=1.0)
    app.handle_key(ord("s"))
    for _ in range(3):
        tick(app, clock)
    steps = app.session.step_count
    app.handle_key(ord("x"))
    for _ in range(100):  # ten seconds of darkness
        tick(app, clock)
    assert app.session.step_count == steps
    app.handle_key(ord("x"))
    for _ in range(5):  # half a second back: still inside the scene's full interval
        tick(app, clock)
    assert app.session.step_count == steps
    for _ in range(10):
        tick(app, clock)
    assert app.session.step_count == steps + 1


def test_a_missing_frame_shows_a_starting_message_not_a_crash():
    app, det, clock, seen = build()
    app.step(None)
    assert seen[-1]["text"] == STARTING_TEXT and seen[-1]["camera_on"] is True
    assert det.calls == 0


def test_the_x_key_does_nothing_for_video_files():
    app, *_ = build()
    app._camera_pausable = False
    app.handle_key(ord("x"))
    assert not app.camera_paused


def test_switching_the_camera_is_logged_in_telemetry(tmp_path):
    app, det, clock, seen = build(telemetry=tmp_path / "logs")
    app.handle_key(ord("x"))
    clock.t += 2
    app.handle_key(ord("x"))
    app._close_telemetry(clock.t)
    rows = [json.loads(line) for line in app.telemetry.path.read_text().splitlines()]
    kinds = [r["e"] for r in rows]
    assert kinds.index("camera_off") < kinds.index("camera_on")


def test_study_mode_still_tells_the_participant_the_camera_state():
    app, det, clock, seen = build(study=True)
    tick(app, clock)
    app.handle_key(ord("x"))
    tick(app, clock)
    assert seen[-2]["camera_on"] is True and seen[-1]["camera_on"] is False
    assert seen[-1]["debug"] is False  # diagnostics hidden, camera status never


# ---------------------------------------------------------------- the indicator itself
def indicator_pixel(camera_on, debug):
    win = GameWindow(960, 540)
    out = win.render(None, "neutral", 0.0, None, True, "x", [], debug=debug, camera_on=camera_on)
    return out[19, 960 // 2 - 60].tolist()  # the middle of the dot


def test_the_camera_dot_is_red_when_on_grey_when_off_and_visible_in_study_mode():
    for debug in (True, False):
        assert indicator_pixel(True, debug) == [0, 0, 230]  # red (BGR)
        assert indicator_pixel(False, debug) == [150, 150, 150]


# ---------------------------------------------------------------- really releasing the camera
class Cam:
    realtime = True
    fps = 30.0
    finished = False

    def __init__(self):
        self.open_now = True
        self.releases = self.reopens = 0
        self.n = 0

    def read(self):
        import time

        time.sleep(0.003)
        if not self.open_now:
            return False, None
        self.n += 1
        return True, np.full((60, 80, 3), self.n % 255, dtype=np.uint8)

    def release(self):
        self.releases += 1
        self.open_now = False

    def reopen(self):
        self.reopens += 1
        self.open_now = True
        return True


def wait_for(cond, timeout=3.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_pipeline_pause_releases_the_device_and_resume_reopens_it():
    cam = Cam()
    pipe = Pipeline(cam, Det(), hz=100, max_width=0)
    pipe.start()
    assert wait_for(lambda: pipe.frame() is not None and pipe.result() is not None)
    pipe.pause()
    assert wait_for(lambda: cam.releases == 1)  # the device itself is released
    assert wait_for(lambda: pipe.frame() is None)  # and no stale picture is kept
    assert pipe.paused and pipe.camera_ok  # switching off is not a "lost camera"
    reads = cam.n
    import time

    time.sleep(0.1)
    assert cam.n == reads and cam.releases == 1  # nothing is read while off
    pipe.resume()
    assert pipe.result() is None  # nothing from before the pause survives
    assert wait_for(lambda: cam.reopens == 1 and pipe.frame() is not None)
    assert wait_for(lambda: pipe.result() is not None)
    assert pipe.stop()


def test_pipeline_can_be_stopped_while_the_camera_is_off():
    cam = Cam()
    pipe = Pipeline(cam, Det(), hz=100, max_width=0)
    pipe.start()
    assert wait_for(lambda: pipe.frame() is not None)
    pipe.pause()
    assert wait_for(lambda: cam.releases == 1)
    assert pipe.stop() is True and not pipe.grabber.alive


class ScriptedWindow:
    """Stub cv2's window functions; send keys at chosen frame counts."""

    def __init__(self, script):
        self.script, self.shown, self.sent = script, 0, set()

    def __enter__(self):
        self.orig = (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, cv2.getWindowProperty)
        cv2.imshow = lambda title, img: setattr(self, "shown", self.shown + 1)

        def wait(ms):
            key = self.script.get(self.shown)
            if key is not None and self.shown not in self.sent:
                self.sent.add(self.shown)
                return key
            return 0xFF

        cv2.waitKey = wait
        cv2.destroyAllWindows = lambda: None
        cv2.getWindowProperty = lambda title, which: 1.0
        return self

    def __exit__(self, *exc):
        cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, cv2.getWindowProperty = self.orig


def worker_threads():
    return [t for t in threading.enumerate() if t.name in ("grabber", "perception")]


@pytest.mark.parametrize("threaded", [True, False])
def test_the_real_loops_release_and_reopen_the_camera_on_x(threaded):
    cam = Cam()
    app = App(Config(backend="haar", calibrate_on_start=False, threaded=threaded), detector=Det())
    original, app_module.STARTUP_TIMEOUT_S = app_module.STARTUP_TIMEOUT_S, 0.2
    try:
        # off at frame 5, off for ~1.3 s (longer than the startup timeout), on at 45, quit at 90
        with ScriptedWindow({5: ord("x"), 45: ord("x"), 90: ord("q")}) as win:
            loop = app._loop_threaded if threaded else app._loop_sync
            assert loop(cam) == 0  # resuming must not be mistaken for a camera that never started
    finally:
        app_module.STARTUP_TIMEOUT_S = original
    assert win.shown >= 90
    assert cam.releases >= 1 and cam.reopens >= 1
    assert not app.camera_paused and worker_threads() == []


# ---------------------------------------------------------------- demo recording
def frames_in(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_record_writes_the_composed_window_to_a_video(tmp_path):
    out = tmp_path / "demo" / "clip.avi"
    app, det, clock, seen = build(record=out)
    canvas = np.zeros((app.cfg.height, app.cfg.width, 3), dtype=np.uint8)
    for i in range(12):
        canvas[:] = i * 10
        app._write_recording(canvas)
    app._shutdown()
    assert out.exists() and frames_in(out) == 12


def test_no_recording_unless_asked_and_a_bad_path_does_not_crash(tmp_path):
    app, *_ = build()
    app._write_recording(np.zeros((540, 960, 3), dtype=np.uint8))
    assert app._recorder is None
    bad = tmp_path / "file.txt" / "clip.avi"
    (tmp_path / "file.txt").write_text("a file, not a folder")
    app2, *_ = build(record=bad)
    for _ in range(3):
        app2._write_recording(np.zeros((540, 960, 3), dtype=np.uint8))  # must not raise
    assert app2._recording_failed


def test_the_window_loop_records_every_shown_frame(tmp_path):
    out = tmp_path / "clip.avi"
    app = App(
        Config(backend="haar", calibrate_on_start=False, threaded=False, record=out), detector=Det()
    )
    with ScriptedWindow({20: ord("q")}) as win:
        assert app._loop_sync(Cam()) == 0
    app._shutdown()
    assert frames_in(out) == win.shown >= 20


def test_video_file_source_still_works_with_the_new_loops(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (80, 60))
    for i in range(20):
        w.write(np.full((60, 80, 3), i * 10, dtype=np.uint8))
    w.release()
    app = App(Config(backend="haar", calibrate_on_start=False, video=path), detector=Det())
    src = VideoFileSource(path, realtime=True)
    assert src.open()
    with ScriptedWindow({}) as win:
        assert app._loop_threaded(src) == 0
    assert win.shown >= 1 and worker_threads() == []
