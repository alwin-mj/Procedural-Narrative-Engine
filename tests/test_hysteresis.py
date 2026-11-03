import random

import pytest

from narrative_engine.perception.smoothing import Ema, EmotionSmoother, HysteresisGate


def avg(**kw):
    d = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
    d.update(kw)
    return d


def test_gate_holds_until_margin_and_hold_frames_met():
    g = HysteresisGate(margin=0.1, hold_frames=3)
    assert g.update("neutral", avg(neutral=0.6)) == "neutral"
    close = avg(neutral=0.5, happy=0.55)  # inside margin
    for _ in range(10):
        assert g.update("happy", close) == "neutral"
    clear = avg(neutral=0.3, happy=0.6)
    assert g.update("happy", clear) == "neutral"
    assert g.update("happy", clear) == "neutral"
    assert g.update("happy", clear) == "happy"  # third consecutive frame


def test_gate_pending_resets_when_candidate_changes():
    g = HysteresisGate(margin=0.1, hold_frames=3)
    g.update("neutral", avg(neutral=0.6))
    a = avg(neutral=0.2, happy=0.5, sad=0.5)
    g.update("happy", a)
    g.update("happy", a)
    g.update("sad", a)  # different candidate restarts the count
    assert g.update("happy", a) == "neutral"


def test_gate_validation_and_reset():
    with pytest.raises(ValueError):
        HysteresisGate(hold_frames=0)
    g = HysteresisGate()
    g.update("sad", avg(sad=1.0))
    g.reset()
    assert g.current is None


def test_ema_converges_and_resets():
    e = Ema(alpha=0.5)
    for _ in range(20):
        e.update(1.0)
    assert e.value == pytest.approx(1.0, abs=1e-3)
    e.reset()
    assert e.value == 0.0
    with pytest.raises(ValueError):
        Ema(alpha=0.0)


def _flips(sequence, use_gate):
    sm = EmotionSmoother(10, 6)
    gate = HysteresisGate(0.08, 3)
    moods = []
    for scores in sequence:
        cand, _, mean = sm.update(scores, max(scores, key=scores.get), max(scores.values()))
        moods.append(gate.update(cand, mean) if use_gate else cand)
    return sum(1 for a, b in zip(moods, moods[1:]) if a != b)


def test_gate_reduces_flicker_on_noisy_input():
    rng = random.Random(7)
    seq = []
    for _ in range(300):  # genuinely ambiguous face: happy and neutral nearly tied
        h = 0.45 + rng.uniform(-0.12, 0.12)
        seq.append(avg(happy=h, neutral=1 - h))
    without, with_gate = _flips(seq, False), _flips(seq, True)
    assert with_gate < without
    assert with_gate <= without * 0.6
