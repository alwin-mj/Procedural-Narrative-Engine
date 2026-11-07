import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from narrative_engine import telemetry as tm
from narrative_engine.app import App
from narrative_engine.config import Config, config_from_args
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import VideoFileSource

BOX = (50, 40, 80, 80)
NOW = datetime(2026, 11, 5, 10, 0, 0)


def det(mood="happy", v=0.5, a=0.2, box=BOX):
    scores = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
    scores[mood] = 1.0
    return Detection(mood, 0.9, box, scores, v, a)


class ScriptedDetector:
    """A viewer whose affect changes over time, so routes depend on it."""

    supports_calibration = False
    calibrated = False
    active_backend = "fake"

    def __init__(self, face_gap=None):
        self.n, self.face_gap = 0, face_gap

    def detect_stable(self, frame):
        self.n += 1
        if self.face_gap and self.face_gap[0] <= self.n < self.face_gap[1]:
            return det("neutral", 0.0, 0.0, box=None)
        v = 0.6 if (self.n // 40) % 2 == 0 else -0.4
        a = 0.35 if (self.n // 25) % 2 == 0 else 0.05
        return det("happy" if v > 0.3 else "sad", v, a)


def write_video(path, frames=300, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for _ in range(frames):
        w.write(np.full((120, 160, 3), 50, dtype=np.uint8))
    w.release()


def live_run(tmp_path, detector=None, **cfg_kw):
    video = tmp_path / "v.avi"
    write_video(video)
    cfg = Config(
        backend="haar",
        video=video,
        headless=True,
        calibrate_on_start=False,
        advance_interval_s=2.0,
        telemetry=tmp_path / "logs",
        participant="P01",
        seed=3,
        **cfg_kw,
    )
    app = App(cfg, detector=detector or ScriptedDetector())
    return app.run_headless(VideoFileSource(video))


def events(path):
    return [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines()]


# ---------------------------------------------------------------- writer / reader
def test_writer_creates_a_session_file_with_a_header(tmp_path):
    meta = {"participant": "P 01/../x", "mode": "adaptive"}
    w = tm.TelemetryWriter(tmp_path / "logs", meta, NOW)
    w.write("tick", 1.23456, **tm.tick_record(det()))
    w.close()
    assert w.path.name == "session_20261105_100000_P01x.jsonl"  # unsafe characters stripped
    rows = events(w.path)
    assert rows[0]["e"] == "session_start" and rows[0]["schema"] == tm.SCHEMA
    assert rows[1]["e"] == "tick" and rows[1]["t"] == 1.235 and rows[1]["s"][0] == 1.0
    assert rows[1]["b"] == list(BOX)


def test_two_sessions_in_the_same_second_do_not_overwrite_each_other(tmp_path):
    a = tm.TelemetryWriter(tmp_path, {"participant": "P1"}, NOW)
    b = tm.TelemetryWriter(tmp_path, {"participant": "P1"}, NOW)
    assert a.path != b.path and a.path.exists() and b.path.exists()
    a.close()
    b.close()


def test_a_failing_disk_disables_logging_instead_of_crashing(tmp_path):
    w = tm.TelemetryWriter(tmp_path, {}, NOW)

    class Broken:
        def write(self, text):
            raise OSError("disk full")

        def close(self):
            pass

    w._f = Broken()
    w.write("tick", 1.0)  # must not raise
    assert w._f is None
    w.write("tick", 2.0)  # and stays quiet afterwards
    w.close()


def test_read_session_roundtrip_and_accessors(tmp_path):
    w = tm.TelemetryWriter(tmp_path, {"mode": "fixed"}, NOW)
    w.write("tick", 0.0, **tm.tick_record(det()))
    w.write("scene", 1.0, id="market")
    w.write("session_end", 2.0, ending="radiant")
    w.close()
    s = tm.read_session(w.path)
    assert s.meta["mode"] == "fixed" and len(s.ticks) == 1 and len(s.scenes) == 1
    assert s.end["ending"] == "radiant"


def test_read_session_rejects_bad_files_and_tolerates_a_cut_off_tail(tmp_path):
    with pytest.raises(ValueError):
        tm.read_session(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tm.read_session(bad)
    nohead = tmp_path / "nohead.jsonl"
    nohead.write_text('{"e":"tick","t":0}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        tm.read_session(nohead)
    future = tmp_path / "future.jsonl"
    future.write_text('{"e":"session_start","t":0,"schema":99}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        tm.read_session(future)
    cut = tmp_path / "cut.jsonl"
    head = '{"e":"session_start","t":0,"schema":1}\n'
    cut.write_text(head + '{"e":"tick","t":1,"m":"ha', encoding="utf-8")
    assert tm.read_session(cut).ticks == []  # a crash mid-write loses only the last line


def test_tick_roundtrip_through_a_detection():
    original = det("sad", -0.4, 0.1)
    back = tm.tick_to_detection(tm.tick_record(original))
    assert back.emotion == "sad" and back.box == BOX and back.valence == -0.4
    assert tm.tick_to_detection(tm.tick_record(det(box=None))).box is None


def test_replay_detector_and_source_follow_the_recording():
    ticks = [{"t": i * 0.5, **tm.tick_record(det("happy" if i < 2 else "sad"))} for i in range(4)]
    d = tm.ReplayDetector(ticks)
    moods = [d.detect_stable(None).emotion for _ in range(6)]
    assert moods == ["happy"] * 2 + ["sad"] * 4  # after the end it holds the last reading
    src = tm.ReplaySource(ticks)
    assert src.open() and src.duration == 1.5 and abs(src.fps - 2.0) < 1e-9
    times = []
    while True:
        ok, frame = src.read()
        if not ok:
            break
        times.append(src.current_time)
        assert frame.shape == (48, 64, 3)
    assert times == [0.0, 0.5, 1.0, 1.5] and src.finished
    with pytest.raises(ValueError):
        tm.ReplayDetector([])
    assert tm.ReplaySource([]).open() is False


# ---------------------------------------------------------------- inside the app
def test_a_live_run_logs_every_kind_of_event(tmp_path):
    report = live_run(tmp_path)
    rows = events(report["telemetry"])
    kinds = [r["e"] for r in rows]
    assert kinds[0] == "session_start" and kinds[-1] == "session_end"
    for k in ("start", "scene", "tick", "ending"):
        assert k in kinds
    head = rows[0]
    assert head["participant"] == "P01" and head["seed"] == 3 and head["mode"] == "adaptive"
    scenes = [r for r in rows if r["e"] == "scene"]
    assert scenes[0]["reason"] == "start" and all("reason" in s for s in scenes)
    assert [s["to"] for s in scenes if s["to"]] == report["route"]  # log and report agree
    assert rows[-1]["route"] == report["route"] and rows[-1]["ending"] == report["ending"]
    assert rows[-1]["metrics"]["detections"] == report["frames"]
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)


def test_no_video_or_image_data_is_ever_logged(tmp_path):
    report = live_run(tmp_path)
    text = open(report["telemetry"], encoding="utf-8").read()
    assert len(text) < 400_000 and "png" not in text.lower() and "base64" not in text.lower()
    for row in events(report["telemetry"]):
        assert not any(isinstance(v, (bytes, bytearray)) for v in row.values())


def test_telemetry_rate_limit(tmp_path):
    def ticks(directory, hz):
        path = live_run(directory, telemetry_hz=hz)["telemetry"]
        return len([e for e in events(path) if e["e"] == "tick"])

    sparse_dir = tmp_path / "b"
    sparse_dir.mkdir()
    every, sparse = ticks(tmp_path, 0), ticks(sparse_dir, 2)
    assert sparse < every / 3


def test_replay_reproduces_the_live_route_exactly(tmp_path):
    report = live_run(tmp_path)
    cfg = Config(
        backend="haar",
        headless=True,
        replay=Path(report["telemetry"]),
        advance_interval_s=2.0,
        calibrate_on_start=False,
    )
    app = App(cfg)
    again = app.run_headless(app._make_source())
    assert again["route"] == report["route"] and again["moods"] == report["moods"]
    assert again["ending"] == report["ending"] and again["decisions"] == report["decisions"]


def test_replay_lets_different_policies_face_the_same_viewer(tmp_path):
    report = live_run(tmp_path)
    routes = {}
    for mode in ("adaptive", "fixed"):
        cfg = Config(
            backend="haar",
            headless=True,
            replay=Path(report["telemetry"]),
            advance_interval_s=2.0,
            director_mode=mode,
            calibrate_on_start=False,
        )
        app = App(cfg)
        routes[mode] = app.run_headless(app._make_source())["route"]
    assert routes["adaptive"] != routes["fixed"]


def test_pause_and_resume_are_logged_when_the_face_disappears(tmp_path):
    report = live_run(tmp_path, detector=ScriptedDetector(face_gap=(30, 90)), no_face_pause_s=2.0)
    rows = events(report["telemetry"])
    pauses = [r for r in rows if r["e"] == "pause"]
    resumes = [r for r in rows if r["e"] == "resume"]
    assert pauses and pauses[0]["why"] == "face" and resumes
    assert pauses[0]["t"] < resumes[0]["t"]


def test_participant_ids_are_validated_and_flags_parse():
    with pytest.raises(ValueError):
        Config(participant="bad id!")
    c = config_from_args(["--telemetry", "--participant", "P07", "--study", "--seed", "5"])
    assert str(c.telemetry) == "logs" and c.participant == "P07" and c.study and c.seed == 5
    assert config_from_args(["--telemetry", "out/here"]).telemetry.as_posix() == "out/here"
    assert config_from_args([]).telemetry is None
    r = config_from_args(["--replay", "s.jsonl"])
    assert r.headless and str(r.replay) == "s.jsonl"
    with pytest.raises(ValueError):
        Config(replay="s.jsonl")  # a replay has no window
    with pytest.raises(ValueError):
        Config(replay="s.jsonl", headless=True, video="v.avi")
    with pytest.raises(ValueError):
        Config(telemetry_hz=-1)


def test_study_mode_hides_everything_that_reveals_the_director():
    class Det(ScriptedDetector):
        pass

    seen = []
    for study in (False, True):
        app = App(Config(backend="haar", study=study, calibrate_on_start=False), detector=Det())
        blank = np.zeros((540, 960, 3), np.uint8)
        app.window.render = lambda *a, _s=seen, **k: _s.append(k) or blank
        app.handle_key(ord("s"))
        app.step(np.zeros((120, 160, 3), dtype=np.uint8))
    normal, blind = seen
    assert normal["debug"] is True and normal["director_note"] and normal["stats"]
    assert blind["debug"] is False and blind["director_note"] == "" and blind["stats"] == ""


def test_seed_makes_the_random_baseline_reproducible(tmp_path):
    def run(seed):
        video = tmp_path / "r.avi"
        write_video(video, frames=200)
        cfg = Config(
            backend="haar",
            video=video,
            headless=True,
            calibrate_on_start=False,
            advance_interval_s=1.0,
            director_mode="random",
            seed=seed,
        )
        return App(cfg, detector=ScriptedDetector()).run_headless(VideoFileSource(video))["route"]

    assert run(11) == run(11)
    assert len({tuple(run(s)) for s in range(8)}) > 1  # and different seeds really differ
