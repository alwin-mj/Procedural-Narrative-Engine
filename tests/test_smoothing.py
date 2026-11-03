import pytest

from narrative_engine.perception.smoothing import EmotionSmoother


def dist(**kw):
    d = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
    d.update(kw)
    return d


def feed(sm, scores, n, dom, conf):
    out = None
    for _ in range(n):
        out = sm.update(scores, dom, conf)
    return out


def test_constant_happy_converges_to_happy():
    sm = EmotionSmoother(10, 6)
    mood, conf, avg = feed(sm, dist(happy=0.9, neutral=0.1), 8, "happy", 0.9)
    assert mood == "happy"
    assert conf == pytest.approx(0.9)
    assert avg["happy"] == pytest.approx(0.9)


def test_single_outlier_frame_does_not_flip_mood():
    sm = EmotionSmoother(10, 6)
    feed(sm, dist(neutral=1.0), 6, "neutral", 0.9)
    mood, _, _ = sm.update(dist(sad=0.9, neutral=0.1), "sad", 0.9)
    assert mood == "neutral"


def test_sustained_change_eventually_flips():
    sm = EmotionSmoother(10, 6)
    feed(sm, dist(neutral=1.0), 6, "neutral", 0.9)
    mood, _, _ = feed(sm, dist(happy=0.9, neutral=0.1), 6, "happy", 0.9)
    assert mood == "happy"


def test_window_cannot_exceed_buffer():
    with pytest.raises(ValueError):
        EmotionSmoother(buffer_size=3, window=6)


def test_reset_clears_history():
    sm = EmotionSmoother(10, 6)
    feed(sm, dist(happy=1.0), 6, "happy", 1.0)
    sm.reset()
    mood, _, avg = sm.update(dist(neutral=1.0), "neutral", 1.0)
    assert mood == "neutral" and avg["happy"] == 0.0
