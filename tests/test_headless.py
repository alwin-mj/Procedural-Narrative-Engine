import json

import cv2
import numpy as np

from narrative_engine.app import App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import VideoFileSource

BOX = (50, 40, 80, 80)


def write_video(path, frames, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8))
    w.release()


class FaceDetector:
    """Always sees a happy face."""

    supports_calibration = False
    calibrated = False
    active_backend = "fake"

    def detect_stable(self, frame):
        s = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
        return Detection("happy", 0.9, BOX, s, 0.6, 0.3)


def make_app(tmp_path, frames, **cfg_kw):
    video = tmp_path / "in.avi"
    write_video(video, frames)
    cfg = Config(backend="haar", video=video, headless=True, calibrate_on_start=False, **cfg_kw)
    return App(cfg, detector=FaceDetector()), video


def test_replay_runs_on_simulated_time_and_advances_the_story(tmp_path):
    app, video = make_app(tmp_path, frames=60, advance_interval_s=1.0)
    report = app.run_headless(VideoFileSource(video))
    assert report["frames"] == 60
    assert report["sim_duration_s"] == 6.0  # 60 frames at 10 fps
    assert len(report["route"]) >= 4  # the story saw ~6 seconds go by
    assert report["wall_s"] < report["sim_duration_s"]  # ...but it took far less than 6 s
    assert report["backend"] == "fake" and report["metrics"]["detections"] == 60


def test_replay_stops_at_the_ending(tmp_path):
    app, video = make_app(tmp_path, frames=600, advance_interval_s=1.0)
    report = app.run_headless(VideoFileSource(video))
    assert report["finished"] and report["ending"] in {e.id for e in app.graph.endings}
    assert report["frames"] < 600
    assert report["decisions"][-1].startswith("final scene")


def test_max_frames_limits_the_run(tmp_path):
    app, video = make_app(tmp_path, frames=60, max_frames=25)
    assert app.run_headless(VideoFileSource(video))["frames"] == 25


def test_replay_is_reproducible(tmp_path):
    def run():
        app, video = make_app(tmp_path, frames=300, advance_interval_s=1.0)
        r = app.run_headless(VideoFileSource(video))
        return r["route"], r["moods"], r["decisions"], r["ending"]

    assert run() == run()


def test_no_face_in_the_video_means_the_story_never_advances(tmp_path):
    video = tmp_path / "blank.avi"
    write_video(video, 40)
    cfg = Config(backend="haar", video=video, headless=True, advance_interval_s=0.5)
    app = App(cfg)  # real Haar detector on flat grey frames: no face
    report = app.run_headless(VideoFileSource(video))
    assert report["frames"] == 40 and len(report["route"]) == 1 and not report["finished"]


def test_run_writes_a_json_report_and_returns_zero(tmp_path):
    out = tmp_path / "reports" / "run.json"
    app, _ = make_app(tmp_path, frames=30, advance_interval_s=1.0, report=out)
    assert app.run() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("frames", "route", "ending", "metrics", "decisions", "mode", "wall_fps"):
        assert key in data
    assert data["frames"] == 30 and data["mode"] == "adaptive"


def test_missing_video_is_a_clean_failure(tmp_path):
    cfg = Config(backend="haar", video=tmp_path / "nope.avi", headless=True)
    assert App(cfg, detector=FaceDetector()).run() == 1


def test_fixed_and_adaptive_modes_can_be_compared_on_the_same_recording(tmp_path):
    video = tmp_path / "in.avi"
    write_video(video, 300)
    routes = {}
    for mode in ("adaptive", "fixed"):
        cfg = Config(
            backend="haar",
            video=video,
            headless=True,
            calibrate_on_start=False,
            advance_interval_s=1.0,
            director_mode=mode,
        )
        app = App(cfg, detector=FaceDetector())
        routes[mode] = app.run_headless(VideoFileSource(video))["route"]
    assert all(r[0] == "wake" and r[-1] == "lighthouse" for r in routes.values())  # both finish
    again = App(
        Config(
            backend="haar",
            video=video,
            headless=True,
            calibrate_on_start=False,
            advance_interval_s=1.0,
            director_mode="fixed",
        ),
        detector=FaceDetector(),
    )
    assert again.run_headless(VideoFileSource(video))["route"] == routes["fixed"]  # deterministic
