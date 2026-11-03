import numpy as np

from narrative_engine.app import App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection

BOX = (50, 40, 80, 80)
FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def scores(mood):
    s = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
    s[mood] = 1.0
    return s


class FakeDetector:
    def __init__(self, supports_calibration=False):
        self.mood, self.conf, self.box = "neutral", 0.9, BOX
        self.supports_calibration = supports_calibration
        self.calibrated = False
        self.active_backend = "fake"
        self.began = self.finished = 0
        self.finish_ok = True

    def detect_stable(self, frame):
        return Detection(self.mood, self.conf, self.box, scores(self.mood), 0.0, 0.0)

    def begin_calibration(self):
        self.began += 1

    def finish_calibration(self):
        self.finished += 1
        self.calibrated = self.finish_ok
        return self.finish_ok


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def build(det=None, **cfg):
    det = det or FakeDetector()
    clock = Clock()
    app = App(Config(backend="haar", calibrate_on_start=False, **cfg), detector=det, clock=clock)
    texts = []

    def record(frame, emo, conf, box, active, text, *a, **k):
        texts.append(text)
        return np.zeros((app.cfg.height, app.cfg.width, 3), dtype=np.uint8)

    app.window.render = record
    return app, det, clock, texts


def run(app, clock, seconds, dt=0.1):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        clock.t += dt
        app.step(FRAME)


def test_flickering_mood_triggers_no_reaction():
    app, det, clock, texts = build()
    app.handle_key(ord("s"))
    for i in range(40):
        det.mood = "happy" if i % 2 else "neutral"  # flips every frame: shorter than dwell time
        clock.t += 0.1
        app.step(FRAME)
    assert not any(t.startswith("A shift") for t in texts)


def test_sustained_mood_shows_reaction_then_it_expires():
    app, det, clock, texts = build(advance_interval_s=999)
    app.handle_key(ord("s"))
    run(app, clock, 0.5)
    det.mood = "happy"
    run(app, clock, 1.5)
    assert any(t.startswith("A shift: you seem happy") for t in texts)
    run(app, clock, 3.0)  # reaction_display_s = 2.0
    assert not texts[-1].startswith("A shift")


def test_mood_change_does_not_postpone_scene_advance():
    app, det, clock, _ = build(advance_interval_s=3.0)
    app.handle_key(ord("s"))
    start_idx = app.story.scene_idx
    run(app, clock, 1.0)
    det.mood = "sad"  # a genuine, sustained shift mid-wait must not reset the timer
    run(app, clock, 2.5)
    assert app.story.scene_idx == start_idx + 1


def test_no_advance_without_a_face_or_confidence():
    app, det, clock, _ = build(advance_interval_s=1.0)
    app.handle_key(ord("s"))
    det.box = None
    run(app, clock, 3.0)
    assert app.story.scene_idx == 0
    det.box, det.conf = BOX, 0.1
    run(app, clock, 3.0)
    assert app.story.scene_idx == 0


def test_calibration_runs_on_first_start_then_not_again():
    det = FakeDetector(supports_calibration=True)
    app, det, clock, texts = build(det, calibration_s=2.0)
    app.cfg = Config(backend="haar", calibrate_on_start=True, calibration_s=2.0)
    app.handle_key(ord("s"))
    run(app, clock, 0.5)
    assert texts[-1].startswith("Hold a neutral face")
    scene_before = app.story.scene_idx
    run(app, clock, 2.0)
    assert det.began == 1 and det.finished == 1
    assert "Calibrated!" in texts
    assert app.story.scene_idx == scene_before  # story paused while calibrating
    app.handle_key(ord("s"))  # restart: already calibrated, no second calibration
    assert det.began == 1


def test_failed_calibration_shows_retry_hint_and_c_key_retries():
    det = FakeDetector(supports_calibration=True)
    det.finish_ok = False
    app, det, clock, texts = build(det, calibration_s=1.0)
    app.handle_key(ord("c"))
    run(app, clock, 1.5)
    assert any("retry" in t.lower() for t in texts)
    app.handle_key(ord("c"))
    assert det.began == 2


def test_calibration_key_without_support_shows_notice():
    app, det, clock, texts = build(FakeDetector(supports_calibration=False))
    app.handle_key(ord("c"))
    run(app, clock, 0.2)
    assert det.began == 0 and "MediaPipe" in texts[-1]


def test_quit_key():
    app, *_ = build()
    assert app.handle_key(ord("q")) is False
