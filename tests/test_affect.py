import pytest

from narrative_engine.perception.affect import (
    baseline_from_samples,
    circumplex_va,
    estimate_affect,
    extract_features,
    remove_baseline,
    scores_from_affect,
)

SMILE = {
    "mouthSmileLeft": 0.7,
    "mouthSmileRight": 0.7,
    "cheekSquintLeft": 0.4,
    "cheekSquintRight": 0.4,
}
FROWN = {"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5, "browDownLeft": 0.4, "browDownRight": 0.4}
SURPRISE = {
    "eyeWideLeft": 0.6,
    "eyeWideRight": 0.6,
    "browInnerUp": 0.6,
    "browOuterUpLeft": 0.5,
    "browOuterUpRight": 0.5,
    "jawOpen": 0.5,
}


def top(blend, baseline=None):
    v, a = estimate_affect(extract_features(blend), baseline)
    scores = scores_from_affect(v, a)
    return max(scores, key=scores.get), v, a, scores


def test_blank_face_is_neutral():
    mood, v, a, scores = top({})
    assert mood == "neutral" and v == 0.0 and a == 0.0
    assert scores["neutral"] > 0.9


def test_smile_frown_surprise_map_to_expected_moods():
    assert top(SMILE)[0] == "happy"
    assert top(FROWN)[0] == "sad"
    assert top(SURPRISE)[0] == "uplift"


def test_valence_sign_and_arousal_range():
    _, v_smile, _, _ = top(SMILE)
    _, v_frown, _, _ = top(FROWN)
    _, _, a_surprise, _ = top(SURPRISE)
    assert v_smile > 0.5 and v_frown < -0.5
    assert 0.0 <= a_surprise <= 1.0


def test_scores_are_a_distribution():
    for v, a in [(0, 0), (1, 0), (-1, 0), (0, 1), (0.5, 0.5), (-0.9, 0.9)]:
        s = scores_from_affect(v, a)
        assert sum(s.values()) == pytest.approx(1.0)
        assert all(x >= 0 for x in s.values())


def test_baseline_removes_a_resting_expression():
    resting = extract_features(SMILE)  # someone whose neutral face looks like a smile
    baseline = baseline_from_samples([resting] * 5)
    mood, v, a, _ = top(SMILE, baseline)
    assert mood == "neutral" and v == pytest.approx(0.0)


def test_baseline_still_allows_a_bigger_expression():
    resting = extract_features({"mouthSmileLeft": 0.3, "mouthSmileRight": 0.3})
    baseline = baseline_from_samples([resting] * 5)
    big = {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9}
    assert top(big, baseline)[0] == "happy"


def test_remove_baseline_is_noop_without_baseline_and_never_negative():
    f = extract_features(SMILE)
    assert remove_baseline(f, None) == f
    out = remove_baseline(f, {k: 1.0 for k in f})
    assert all(0.0 <= x <= 1.0 for x in out.values())


def test_baseline_needs_samples():
    with pytest.raises(ValueError):
        baseline_from_samples([])


def test_circumplex_orders_emotions_sensibly():
    v_happy, _ = circumplex_va({"happy": 1.0})
    v_sad, a_sad = circumplex_va({"sad": 1.0})
    _, a_angry = circumplex_va({"angry": 1.0})
    assert v_happy > 0 > v_sad
    assert a_angry > a_sad
    assert circumplex_va({}) == (0.0, 0.0)
    assert circumplex_va({"unknown": 1.0}) == (0.0, 0.0)
    # DeepFace-style 0..100 values give the same answer as probabilities
    assert circumplex_va({"happy": 90, "neutral": 10}) == pytest.approx(
        circumplex_va({"happy": 0.9, "neutral": 0.1})
    )
