from narrative_engine.director import choose_advance_emotion as choose


def test_empty_scores_fall_back():
    assert choose({}, None, None) == "neutral"
    assert choose({}, "happy", None) == "happy"


def test_clear_happy_wins():
    scores = {"happy": 0.8, "uplift": 0.05, "sad": 0.05, "neutral": 0.1}
    assert choose(scores, None, None) == "happy"


def test_happy_bias_steers_neutral_to_happy():
    scores = {"happy": 0.3, "uplift": 0.0, "sad": 0.2, "neutral": 0.5}
    assert choose(scores, None, None, happy_bias=True) == "happy"
    assert choose(scores, None, None, happy_bias=False) == "neutral"


def test_different_top_wins_over_last_choice():
    scores = {"happy": 0.1, "uplift": 0.7, "sad": 0.1, "neutral": 0.1}
    assert choose(scores, "sad", None, happy_bias=False) == "uplift"


def test_repeated_top_switches_to_mood_that_grew():
    last = {"happy": 0.1, "uplift": 0.1, "sad": 0.6, "neutral": 0.2}
    now = {"happy": 0.1, "uplift": 0.3, "sad": 0.5, "neutral": 0.1}
    assert choose(now, "sad", last, happy_bias=False) == "uplift"


def test_close_runner_up_takes_over_when_nothing_grew():
    scores = {"happy": 0.0, "uplift": 0.0, "sad": 0.5, "neutral": 0.5}
    assert choose(scores, "sad", {"sad": 0.9}, happy_bias=False) == "neutral"
