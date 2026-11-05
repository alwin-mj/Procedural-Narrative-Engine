import numpy as np

from narrative_engine.app import CAMERA_LOST_TEXT, NO_FACE_TEXT, PROMPT_TEXT, App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import PerceptionResult

BOX = (50, 40, 80, 80)
FRAME = np.zeros((120, 160, 3), dtype=np.uint8)
SCORES = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class NeverCalled:
    """Threaded steps must not touch the detector on the UI thread."""

    supports_calibration = False
    calibrated = False
    active_backend = "worker"

    def detect_stable(self, frame):
        raise AssertionError("UI thread ran the detector")


def result(clock, fid, box=BOX, v=0.8, a=0.4, age=0.0):
    d = Detection("happy", 0.9, box, dict(SCORES), v, a)
    return PerceptionResult(d, clock.t - age, clock.t - age, fid)


def build(**cfg):
    clock = Clock()
    app = App(Config(backend="haar", calibrate_on_start=False, **cfg),
              detector=NeverCalled(), clock=clock)
    seen = []

    def record(frame, emo, conf, box, active, text, *a, **k):
        seen.append({"text": text, "box": box, "emo": emo, **k})
        return np.zeros((app.cfg.height, app.cfg.width, 3), dtype=np.uint8)

    app.window.render = record
    return app, clock, seen


def tick(app, clock, res, seconds=0.1, camera_ok=True):
    clock.t += seconds
    return app.step(FRAME, res, camera_ok)


def test_threaded_step_never_runs_the_detector_on_the_ui_thread():
    app, clock, seen = build()
    tick(app, clock, None)  # worker has produced nothing yet
    tick(app, clock, result(clock, 1))
    assert seen[0]["text"] == PROMPT_TEXT and seen[1]["emo"] == "happy"


def test_stale_results_decay_valence_but_keep_the_last_mood():
    app, clock, seen = build()
    r = result(clock, 1, v=0.8, a=0.6)
    tick(app, clock, r)
    fresh_v = seen[-1]["valence"]
    tick(app, clock, r, seconds=1.5)  # no new reading for ~1.6 s
    assert seen[-1]["valence"] < fresh_v * 0.6
    assert seen[-1]["emo"] == "happy"  # keeps the last stable mood
    assert "LAGGING" in seen[-1]["stats"]
    tick(app, clock, r, seconds=3.0)
    assert seen[-1]["box"] is None  # long stall: the face is treated as gone


def test_motion_history_and_scene_affect_update_once_per_new_reading():
    app, clock, seen = build()
    calls = []
    real = app.motion.update
    app.motion.update = lambda box: calls.append(box) or real(box)
    r = result(clock, 7)
    for _ in range(10):  # the renderer runs 10 frames on the same detector result
        tick(app, clock, r, seconds=0.03)
    assert len(calls) == 1 and len(app.history) == 1
    tick(app, clock, result(clock, 8))
    assert len(calls) == 2 and len(app.history) == 2


def test_losing_the_face_pauses_the_story_and_says_so():
    app, clock, seen = build(advance_interval_s=1.0, no_face_pause_s=2.0)
    app.handle_key(ord("s"))
    fid = 1
    for _ in range(5):  # normal play
        fid += 1
        tick(app, clock, result(clock, fid))
    step_before = app.session.step_count
    for _ in range(30):  # 3 s with no face
        fid += 1
        tick(app, clock, result(clock, fid, box=None))
    assert seen[-1]["text"] == NO_FACE_TEXT
    assert app.session.step_count == step_before  # nothing advanced while away
    # the viewer returns: the scene gets its full time before advancing
    fid += 1
    tick(app, clock, result(clock, fid))
    assert seen[-1]["text"] != NO_FACE_TEXT
    assert app.session.step_count == step_before
    for _ in range(5):
        fid += 1
        tick(app, clock, result(clock, fid))
    assert app.session.step_count == step_before  # still within the 1 s hold
    for _ in range(8):
        fid += 1
        tick(app, clock, result(clock, fid))
    assert app.session.step_count == step_before + 1


def test_camera_disconnect_shows_a_message_and_pauses_then_resumes():
    app, clock, seen = build(advance_interval_s=1.0)
    app.handle_key(ord("s"))
    fid = 1
    for _ in range(3):
        fid += 1
        tick(app, clock, result(clock, fid))
    step_before = app.session.step_count
    for _ in range(40):
        fid += 1
        tick(app, clock, result(clock, fid), camera_ok=False)
    assert seen[-1]["text"] == CAMERA_LOST_TEXT
    assert app.session.step_count == step_before
    for _ in range(25):
        fid += 1
        tick(app, clock, result(clock, fid), camera_ok=True)
    assert seen[-1]["text"] != CAMERA_LOST_TEXT
    assert app.session.step_count > step_before  # play continues after reconnecting


def test_shutdown_is_idempotent_and_closes_everything_once():
    class Closable(NeverCalled):
        closed = 0

        def close(self):
            Closable.closed += 1

    app, clock, _ = build()
    app.detector = Closable()
    app._shutdown()
    app._shutdown()  # safe to call twice
    assert Closable.closed == 1
