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
    start_idx = app.session.step_count
    run(app, clock, 1.0)
    det.mood = "sad"  # a genuine, sustained shift mid-wait must not reset the timer
    run(app, clock, 2.5)
    assert app.session.step_count == start_idx + 1


def test_no_advance_without_a_face_or_confidence():
    app, det, clock, _ = build(advance_interval_s=1.0)
    app.handle_key(ord("s"))
    det.box = None
    run(app, clock, 3.0)
    assert app.session.step_count == 0
    det.box, det.conf = BOX, 0.1
    run(app, clock, 3.0)
    assert app.session.step_count == 0


def test_calibration_runs_on_first_start_then_not_again():
    det = FakeDetector(supports_calibration=True)
    app, det, clock, texts = build(det, calibration_s=2.0)
    app.cfg = Config(backend="haar", calibrate_on_start=True, calibration_s=2.0)
    app.handle_key(ord("s"))
    run(app, clock, 0.5)
    assert texts[-1].startswith("Hold a neutral face")
    scene_before = app.session.step_count
    run(app, clock, 2.0)
    assert det.began == 1 and det.finished == 1
    assert "Calibrated!" in texts
    assert app.session.step_count == scene_before  # story paused while calibrating
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


# ---------------------------------------------------------------- Phase 4: director integration
import json  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from narrative_engine.director.realizer import (  # noqa: E402
    BeatRequest,
    LLMRealizer,
    RealizerWorker,
)


def det_with(app_det, valence=0.0, arousal=0.0):
    app_det.valence, app_det.arousal = valence, arousal


def play_to_end(app, clock, det, max_seconds=200):
    app.handle_key(ord("s"))
    t = 0.0
    while not app.session.finished and t < max_seconds:
        run(app, clock, 1.0)
        t += 1.0
    return app.session


def test_full_playthrough_reaches_an_ending_and_logs_decisions():
    app, det, clock, texts = build(advance_interval_s=0.5)
    s = play_to_end(app, clock, det)
    assert s.finished and s.ending_id in {e.id for e in app.graph.endings}
    assert app.last_decision is not None and app.last_decision.to_id is None
    assert texts[-1] == s.ending_text()
    # restart works after an ending
    app.handle_key(ord("s"))
    assert not app.session.finished and app.session.step_count == 0


def test_hud_note_shows_scene_and_reason():
    app, det, clock, _ = build(advance_interval_s=0.5)
    notes = []
    orig = app.window.render

    def rec(*a, **k):
        notes.append(k.get("director_note", ""))
        return orig(*a, **k) if False else np.zeros((app.cfg.height, app.cfg.width, 3), np.uint8)

    app.window.render = rec
    app.handle_key(ord("s"))
    run(app, clock, 2.0)
    assert any(n.startswith("scene: ") and "tension" in n for n in notes)
    assert any("->" in n for n in notes)  # a decision reason has appeared


def test_fixed_mode_always_shows_neutral_and_never_branches_by_affect():
    app, det, clock, texts = build(advance_interval_s=0.5, director_mode="fixed")
    det.mood = "happy"
    s = play_to_end(app, clock, det)
    # with a flat face no conditional edge applies, so 'fixed' follows the unconditional default
    first_edges = [
        next(e.to for e in app.graph.nodes[n].edges if e.when.always) for n in s.state.visited[:-1]
    ]
    assert s.state.visited[1:] == first_edges
    assert set(s.state.moods) == {"neutral"}


def test_random_mode_is_reproducible_with_a_seeded_rng():
    def route(seed):
        det = FakeDetector()
        clock = Clock()
        cfg = Config(
            backend="haar", calibrate_on_start=False, advance_interval_s=0.5, director_mode="random"
        )
        app = App(cfg, detector=det, clock=clock, rng=random.Random(seed))
        blank = np.zeros((cfg.height, cfg.width, 3), np.uint8)
        app.window.render = lambda *a, **k: blank
        return play_to_end(app, clock, det).state.visited

    assert route(3) == route(3)
    assert len({tuple(route(s)) for s in range(12)}) > 1


def test_mood_bias_off_by_default_keeps_neutral_variant_for_neutral_face():
    app, det, clock, _ = build(advance_interval_s=0.5)
    det.mood = "neutral"
    s = play_to_end(app, clock, det)
    assert set(s.state.moods) == {"neutral"}


# ---------------------------------------------------------------- Phase 4: LLM realizer
class ScriptedRealizer:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def realize(self, req: BeatRequest):
        self.calls.append(req)
        if self.fail:
            raise OSError("down")
        return f"Arin, retold: {req.authored[:40]}"


def wait(app):
    assert app.worker.wait_idle(3.0)


def test_llm_text_replaces_authored_text_when_ready_and_never_blocks():
    realizer = ScriptedRealizer()
    app, det, clock, texts = build(advance_interval_s=999)
    app.worker = RealizerWorker(realizer)
    app.handle_key(ord("s"))
    app.step(FRAME)
    authored = app.session.get_line("neutral")
    assert texts[-1] == authored  # shown immediately, no waiting for the model
    wait(app)
    app.step(FRAME)
    assert texts[-1].startswith("Arin, retold:")
    app.worker.close()


def test_prefetch_has_next_scene_text_ready_before_advancing():
    realizer = ScriptedRealizer()
    app, det, clock, texts = build(advance_interval_s=999)
    app.worker = RealizerWorker(realizer)
    app.handle_key(ord("s"))
    wait(app)
    prefetched = {r.authored for r in realizer.calls}
    app.handle_key(ord(" "))  # advance
    now_line = app.session.get_line(app._display_mood("neutral"))
    assert now_line in prefetched or any(now_line.startswith(p[:20]) for p in prefetched)
    app.step(FRAME)
    assert texts[-1].startswith("Arin, retold:")  # cache hit on the first frame
    app.worker.close()


def test_llm_failure_falls_back_to_authored_text_and_is_counted():
    realizer = ScriptedRealizer(fail=True)
    app, det, clock, texts = build(advance_interval_s=999)
    app.worker = RealizerWorker(realizer)
    app.handle_key(ord("s"))
    wait(app)
    app.step(FRAME)
    assert texts[-1] == app.session.get_line("neutral")
    st = app.worker.stats()
    assert st["fallbacks"] >= 1 and st["successes"] == 0 and st["fallback_rate"] == 1.0
    app.worker.close()


def test_worker_pauses_after_repeated_failures_and_recovers():
    clock = Clock()
    realizer = ScriptedRealizer(fail=True)
    worker = RealizerWorker(realizer, clock=clock)
    req = BeatRequest(authored="Arin walks along the quiet river bank.", mood="neutral")
    for i in range(3):
        assert worker.submit(("k", i), req)
        assert worker.wait_idle(2.0)
    assert worker.submit(("k", 99), req) is False  # paused
    clock.t += 31
    assert worker.submit(("k", 100), req) is True  # pause expired
    worker.wait_idle(2.0)
    worker.close()


def test_real_http_path_against_a_fake_ollama_server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = "Arin steps into the busy market, grinning at the spices."
            out = json.dumps({"response": reply})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = LLMRealizer("tiny", f"http://127.0.0.1:{server.server_port}", timeout_s=3)
        req = BeatRequest(authored="Arin enters the market.", mood="happy", protagonist="Arin",
                          title="Arin's Day", location="market", recent=("Arin wakes.",))
        text = llm.realize(req)
    finally:
        server.shutdown()
    assert text == "Arin steps into the busy market, grinning at the spices."
    assert seen["path"] == "/api/generate" and seen["body"]["model"] == "tiny"
    assert seen["body"]["stream"] is False
    prompt = seen["body"]["prompt"]
    assert "Arin enters the market." in prompt and "joyful" in prompt


def test_unreachable_server_raises_quickly_and_worker_survives():
    llm = LLMRealizer("tiny", "http://127.0.0.1:9", timeout_s=1, warmup_timeout_s=1)
    worker = RealizerWorker(llm)
    req = BeatRequest(authored="Arin walks along the quiet river bank.", mood="neutral")
    assert worker.submit("k", req)
    assert worker.wait_idle(5.0)
    assert worker.get("k") is None and worker.stats()["fallbacks"] == 1
    worker.close()
