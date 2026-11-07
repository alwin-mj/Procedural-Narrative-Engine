import pytest

from narrative_engine.director.realizer import (
    AuthoredRealizer,
    BeatRequest,
    build_prompt,
    clean_output,
)

GOOD = "Arin wanders into the spice market, where merchants call softly."


def test_clean_output_accepts_good_narration_and_strips_quotes():
    assert clean_output(f'"{GOOD}"', "Arin") == GOOD
    assert clean_output(f"  {GOOD}\n", "Arin") == GOOD


def test_clean_output_rejects_bad_output():
    assert clean_output("", "Arin") is None
    assert clean_output("Too short.", "Arin") is None
    assert clean_output("Sure! Here is the scene you wanted for the story today.", "Arin") is None
    nameless = "A stranger wanders into a market where merchants call softly."
    assert clean_output(nameless, "Arin") is None
    listed = "Arin walks.\n\n- bullet one\n- bullet two and more words here"
    assert clean_output(listed, "Arin") is None
    numbered = "1. Arin walks into the market\n2. Arin buys an apple from a seller"
    assert clean_output(numbered, "Arin") is None
    assert clean_output("**Arin** walks through the market where merchants call.", "Arin") is None


def test_clean_output_trims_long_text_at_a_sentence_boundary():
    long = "Arin walks on. " * 40
    out = clean_output(long, "Arin", max_chars=100)
    assert out is not None and len(out) <= 100 and out.endswith(".")


def test_clean_output_without_protagonist_does_not_require_a_name():
    assert clean_output("A quiet market wakes slowly as merchants arrange their wares.") is not None


def test_prompt_contains_the_facts_and_tone():
    req = BeatRequest(
        authored="He buys a bright orange pouch.",
        mood="sad",
        title="T",
        protagonist="Arin",
        location="market",
        recent=("one", "two", "three", "four"),
    )
    p = build_prompt(req)
    assert "He buys a bright orange pouch." in p and "Arin" in p and "market" in p
    assert "comforting" in p and "four" in p and "one" not in p.split("Story so far:")[1][:30]


def test_prompt_asks_for_the_hero_name_only_when_the_scene_uses_it():
    def beat(text):
        return BeatRequest(authored=text, mood="happy", protagonist="Arin")

    named, unnamed = beat("Arin wakes at the edge of town."), beat("He wakes at the edge of town.")
    assert "Use the name Arin at least once." in build_prompt(named)
    assert "Use the name" not in build_prompt(unnamed)


def test_authored_realizer_returns_the_authored_line():
    assert AuthoredRealizer().realize(BeatRequest(authored="x y z", mood="happy")) == "x y z"


# ------------------------------------------------------- the hero rule, reasons, Ollama check
import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from narrative_engine.director.realizer import (  # noqa: E402
    LLMRealizer,
    OutputRejected,
    RealizerWorker,
    check_ollama,
    check_output,
)

HE_LINE = "He wanders into the market where stalls smell of spice and wood."


def test_hero_name_is_only_required_when_the_authored_line_has_it():
    retelling = "He drifts into the market, drawn in by the warm smell of spice."
    assert check_output(retelling, "Arin", authored=HE_LINE) == (retelling, None)  # "he" is fine
    text, why = check_output(retelling, "Arin", authored="Arin wanders into the market.")
    assert text is None and "Arin" in why  # the line names him, so the retelling must too
    assert check_output(retelling, "Arin")[0] is None  # no authored text given: always required


@pytest.mark.parametrize(
    "bad, expected",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("- one thing\n- another thing", "list"),
        ("Here is the retold scene: Arin walks.", "preamble"),
        ("I'm sorry, but I can't help with that request.", "refusal"),
        ("As an AI language model I cannot do this.", "refusal"),
        ("Too short.", "shorter"),
        ("**Arin** wanders into the busy market place.", "markdown"),
        ("word" * 100, "longer"),
    ],
)
def test_every_rejection_has_a_readable_reason(bad, expected):
    text, why = check_output(bad)
    assert text is None and expected in why


def test_accepted_text_has_no_reason():
    assert check_output("Arin steps into the busy market, grinning at the spices.")[1] is None


class FakeOllama(BaseHTTPRequestHandler):
    models = [{"name": "llama3.2:latest"}]
    reply = "He drifts into the market, drawn by the warm smell of spice and wood."

    def do_GET(self):
        body = json.dumps({"models": self.models}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"response": self.reply}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_check_ollama_reports_reachability_and_missing_models():
    server, host = serve(FakeOllama)
    try:
        ok, msg = check_ollama(host, "llama3.2")
        assert ok and "llama3.2" in msg
        ok, msg = check_ollama(host, "llama3.2:latest")
        assert ok
        ok, msg = check_ollama(host, "mistral")
        assert not ok and "ollama pull mistral" in msg and "llama3.2:latest" in msg
    finally:
        server.shutdown()
    ok, msg = check_ollama("http://127.0.0.1:9", "llama3.2", timeout_s=1)
    assert not ok and "ollama serve" in msg and "not reachable" in msg


def test_llm_realizer_accepts_a_pronoun_retelling_of_a_pronoun_line():
    server, host = serve(FakeOllama)
    try:
        llm = LLMRealizer("tiny", host, timeout_s=5)
        req = BeatRequest(authored=HE_LINE, mood="neutral", protagonist="Arin")
        assert llm.realize(req).startswith("He drifts into the market")
        bad = BeatRequest(
            authored="Arin wanders into the market.", mood="neutral", protagonist="Arin"
        )
        with pytest.raises(OutputRejected, match="hero"):
            llm.realize(bad)  # the scene names Arin; the retelling lost him
    finally:
        server.shutdown()


class RejectingRealizer:
    def __init__(self):
        self.calls = 0

    def realize(self, req):
        self.calls += 1
        raise OutputRejected("lost the hero", "He walks.")


def test_rejected_answers_are_counted_separately_and_never_pause_the_worker():
    worker = RealizerWorker(RejectingRealizer())
    req = BeatRequest(authored="He walks along the quiet river bank.", mood="neutral")
    for i in range(6):  # well past the 3-failure breaker
        assert worker.submit(("k", i), req)
        assert worker.wait_idle(2.0)
    assert worker.submit(("k", 99), req) is True  # still accepting work: the server is fine
    worker.wait_idle(2.0)
    stats = worker.stats()
    assert stats["rejected"] == 7 and stats["errors"] == 0 and stats["successes"] == 0
    assert stats["fallbacks"] == 7 and stats["fallback_rate"] == 1.0
    worker.close()


def test_a_rejection_resets_the_error_streak_because_the_server_clearly_answered():
    class Mixed:
        def __init__(self):
            self.plan = [OSError("down"), OSError("down"), OutputRejected("x"), OSError("down")]

        def realize(self, req):
            raise self.plan.pop(0)

    worker = RealizerWorker(Mixed())
    req = BeatRequest(authored="He walks along the quiet river bank.", mood="neutral")
    for i in range(4):
        worker.submit(("k", i), req)
        worker.wait_idle(2.0)
    assert worker.submit(("k", 9), req) is True  # 2 errors, a rejection, 1 error: no 3 in a row
    worker.wait_idle(2.0)
    worker.close()


def test_stats_include_errors_and_rejections():
    worker = RealizerWorker(AuthoredRealizer())
    assert {"requests", "successes", "errors", "rejected", "fallbacks", "fallback_rate"} <= set(
        worker.stats()
    )
    worker.close()
