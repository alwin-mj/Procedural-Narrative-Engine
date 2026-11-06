import threading
import time

import cv2
import numpy as np

from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import (
    FrameGrabber,
    Metrics,
    PerceptionWorker,
    Pipeline,
    ReconnectingSource,
    VideoFileSource,
    degrade,
    downscale,
    percentile,
    rescale_box,
)

SCORES = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 1.0}


def det(box=(10, 10, 50, 50), v=0.8, a=0.4, emo="happy"):
    return Detection(emo, 0.9, box, dict(SCORES), v, a)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


class FakeSource:
    """A camera whose health the test controls."""

    realtime = True
    fps = 30.0

    def __init__(self, ok=True, reopen_ok=True, delay=0.002, size=(120, 160)):
        self.ok, self.reopen_ok, self.delay, self.size = ok, reopen_ok, delay, size
        self.finished = False
        self.reads = self.reopens = 0
        self.released = False
        self.raise_on_read = False

    def read(self):
        time.sleep(self.delay)
        self.reads += 1
        if self.raise_on_read:
            raise OSError("driver hiccup")
        if not self.ok:
            return False, None
        return True, np.full((*self.size, 3), self.reads % 255, dtype=np.uint8)

    def reopen(self):
        self.reopens += 1
        if self.reopen_ok:
            self.ok = True
        return self.reopen_ok

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, delay=0.0, fail_every=0, box=(10, 10, 50, 50)):
        self.delay, self.fail_every, self.box = delay, fail_every, box
        self.calls = 0
        self.widths = []

    def detect_stable(self, frame):
        self.calls += 1
        self.widths.append(frame.shape[1])
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("model hiccup")
        time.sleep(self.delay)
        return det(self.box)


# ---------------------------------------------------------------- ReconnectingSource
def test_reconnecting_source_marks_camera_lost_then_recovers():
    now = [0.0]
    src = FakeSource(delay=0)
    rs = ReconnectingSource(src, lost_after=3, retry_s=1.0, clock=lambda: now[0])
    assert rs.read() is not None and rs.connected
    src.ok, src.reopen_ok = False, False
    for _ in range(3):
        assert rs.read() is None
    assert not rs.connected  # lost after 3 consecutive failures
    now[0] += 0.5
    rs.read()
    assert src.reopens == 0  # too soon to retry
    now[0] += 1.0
    rs.read()
    assert src.reopens == 1 and not rs.connected  # retried, still down
    src.reopen_ok = True
    now[0] += 1.5
    rs.read()
    assert src.reopens == 2 and rs.connected  # back
    assert rs.read() is not None


def test_reconnecting_source_survives_driver_exceptions():
    src = FakeSource(delay=0)
    src.raise_on_read = True
    rs = ReconnectingSource(src, lost_after=2, retry_s=100)
    assert rs.read() is None and rs.read() is None
    assert not rs.connected


def test_finished_source_is_ended_not_lost():
    src = FakeSource(delay=0)
    src.ok, src.finished = False, True
    rs = ReconnectingSource(src, lost_after=1)
    assert rs.read() is None
    assert rs.ended and rs.connected


# ---------------------------------------------------------------- video source
def write_video(path, frames=20, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), (i * 11) % 255, dtype=np.uint8))
    w.release()


def test_video_source_reads_every_frame_then_finishes(tmp_path):
    path = tmp_path / "v.avi"
    write_video(path, frames=20, fps=10)
    src = VideoFileSource(path)
    assert src.open() and abs(src.fps - 10) < 0.5
    n = 0
    while src.read()[0]:
        n += 1
    assert n == 20 and src.finished
    assert src.reopen() is False
    src.release()


def test_video_source_missing_file_fails_to_open(tmp_path):
    assert VideoFileSource(tmp_path / "nope.avi").open() is False


# ---------------------------------------------------------------- helpers
def test_downscale_and_rescale_box():
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    small, scale = downscale(big, 320)
    assert small.shape == (240, 320, 3) and scale == 0.5
    same, s1 = downscale(big, 0)
    assert same is big and s1 == 1.0
    assert downscale(big, 1000)[1] == 1.0
    assert rescale_box((10, 20, 30, 40), 2.0) == (20, 40, 60, 80)
    assert rescale_box(None, 2.0) is None and rescale_box((1, 2, 3, 4), 1.0) == (1, 2, 3, 4)


def test_degrade_decays_toward_neutral_and_drops_the_face():
    d = det(v=0.8, a=0.6)
    fresh, lag = degrade(d, 0.1, 0.5, 2.0, 1.0)
    assert fresh is d and lag is False
    stale, lag = degrade(d, 1.5, 0.5, 2.0, 1.0)  # one half-life past the threshold
    assert lag and abs(stale.valence - 0.4) < 1e-6 and abs(stale.arousal - 0.3) < 1e-6
    assert stale.box == d.box and stale.emotion == d.emotion  # keeps last stable mood
    gone, lag = degrade(d, 2.5, 0.5, 2.0, 1.0)
    assert gone.box is None and lag and abs(gone.valence - 0.2) < 1e-6


def test_metrics_rates_and_percentiles():
    t = [0.0]
    m = Metrics(clock=lambda: t[0])
    for _ in range(11):
        m.record_frame()
        t[0] += 0.1
    assert abs(m.fps() - 10.0) < 0.01
    for ms in range(1, 101):
        m.record_perception(float(ms), float(ms) * 2)
    snap = m.snapshot()
    assert snap["frames"] == 11 and snap["detections"] == 100
    assert 94 <= snap["inference_ms_p95"] <= 96
    assert snap["latency_ms_p95"] > snap["inference_ms_p95"]
    assert "fps" in m.hud_text()
    assert percentile([], 95) == 0.0


# ---------------------------------------------------------------- threads
def test_grabber_keeps_the_newest_frame_and_stops_cleanly():
    src = FakeSource()
    g = FrameGrabber(ReconnectingSource(src))
    g.start()
    assert wait_for(lambda: g.latest()[1] >= 10)
    frame, fid, _ = g.latest()
    assert frame is not None and fid >= 10
    assert g.stop() is True and not g.alive


def test_grabber_exits_by_itself_when_the_video_ends(tmp_path):
    path = tmp_path / "v.avi"
    write_video(path, frames=5)
    src = VideoFileSource(path)
    assert src.open()
    g = FrameGrabber(ReconnectingSource(src))
    g.start()
    assert wait_for(lambda: g.ended and not g.alive)
    assert g.latest()[1] == 5


def test_slow_detector_drops_old_frames_instead_of_queueing():
    g = FrameGrabber(ReconnectingSource(FakeSource(delay=0.003)))
    d = FakeDetector(delay=0.05)
    w = PerceptionWorker(g, d, hz=1000, max_width=0)
    g.start()
    w.start()
    time.sleep(0.8)
    captured = g.latest()[1]
    res = w.latest()
    assert w.stop() and g.stop()
    assert w.processed <= 17  # ~50 ms per frame
    assert captured > w.processed * 3  # most frames were skipped, not queued
    assert res is not None and captured - res.frame_id < 40  # and the result is recent


def test_detector_rate_is_throttled_to_hz():
    g = FrameGrabber(ReconnectingSource(FakeSource(delay=0.002)))
    w = PerceptionWorker(g, FakeDetector(), hz=20, max_width=0)
    g.start()
    w.start()
    time.sleep(0.6)
    w.stop()
    g.stop()
    assert 5 <= w.processed <= 16  # ~12 expected


def test_frames_are_downscaled_and_boxes_scaled_back():
    g = FrameGrabber(ReconnectingSource(FakeSource(size=(480, 640))))
    d = FakeDetector(box=(10, 10, 50, 50))
    w = PerceptionWorker(g, d, hz=100, max_width=320)
    g.start()
    w.start()
    assert wait_for(lambda: w.latest() is not None)
    res = w.latest()
    w.stop()
    g.stop()
    assert set(d.widths) == {320}  # the detector only ever saw small frames
    assert res.detection.box == (20, 20, 100, 100)  # but the box is in full-frame coordinates


def test_detector_exceptions_do_not_kill_perception():
    g = FrameGrabber(ReconnectingSource(FakeSource()))
    w = PerceptionWorker(g, FakeDetector(fail_every=2), hz=200, max_width=0)
    g.start()
    w.start()
    assert wait_for(lambda: w.processed >= 5 and w.errors >= 2)
    assert w.alive
    assert w.stop() and g.stop()


def test_pipeline_end_to_end_start_stop_and_release():
    src = FakeSource()
    metrics = Metrics()
    pipe = Pipeline(src, FakeDetector(), hz=50, max_width=0, metrics=metrics)
    pipe.start()
    assert wait_for(lambda: pipe.result() is not None and pipe.frame() is not None)
    assert pipe.camera_ok and not pipe.ended
    assert wait_for(lambda: metrics.detections >= 3)
    before = threading.active_count()
    assert pipe.stop() is True
    assert not pipe.grabber.alive and not pipe.worker.alive
    assert src.released
    assert threading.active_count() <= before - 2


def test_pipeline_reports_camera_loss_and_recovery():
    src = FakeSource(reopen_ok=False)
    pipe = Pipeline(src, FakeDetector(), hz=50, max_width=0, lost_after=5, retry_s=0.05)
    pipe.start()
    assert wait_for(lambda: pipe.frame() is not None)
    src.ok = False  # unplug
    assert wait_for(lambda: not pipe.camera_ok)
    assert pipe.frame() is not None  # last good frame is still available for the UI
    src.reopen_ok = True  # plug back in
    assert wait_for(lambda: pipe.camera_ok)
    assert pipe.stop()


# ---------------------------------------------------------------- CameraSource / pacing
from narrative_engine.pipeline import CameraSource  # noqa: E402


class FakeCap:
    def __init__(self, frames):
        self.frames, self.released = list(frames), False

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def isOpened(self):
        return True

    def release(self):
        self.released = True


def test_camera_source_without_a_camera_fails_cleanly():
    cam = CameraSource(index=99)
    assert cam.read() == (False, None)  # not opened yet
    assert cam.open() is False  # no such device
    assert cam.reopen() is False
    cam.release()
    cam.release()  # idempotent


def test_camera_source_mirrors_frames_and_reports_failed_reads():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0] = 255  # bright left edge
    cam = CameraSource(mirror=True)
    cam._cap = FakeCap([frame])
    ok, out = cam.read()
    assert ok and out[:, -1].min() == 255 and out[:, 0].max() == 0  # flipped left-right
    assert cam.read() == (False, None)
    raw = CameraSource(mirror=False)
    raw._cap = FakeCap([frame])
    assert raw.read()[1][:, 0].min() == 255
    cap = FakeCap([])
    raw._cap = cap
    raw.release()
    assert cap.released and raw._cap is None


def test_realtime_video_playback_is_paced_to_the_files_frame_rate(tmp_path):
    path = tmp_path / "v.avi"
    write_video(path, frames=6, fps=50)
    src = VideoFileSource(path, realtime=True)
    assert src.open()
    t0 = time.monotonic()
    while src.read()[0]:
        pass
    assert time.monotonic() - t0 >= 0.08  # 6 frames at 50 fps take at least ~0.1 s
