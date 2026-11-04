import random

import pytest

from narrative_engine.director import (
    Affect,
    DramaManager,
    StoryGraph,
    StorySession,
    TensionCurve,
    load_story,
)
from narrative_engine.config import MOODS


def v(tag):
    return {m: f"{tag}-{m}" for m in MOODS}


def fork_graph(low=0.2, high=0.9):
    """start -> (calm | wild) -> end, so the manager has a real choice."""
    return StoryGraph.from_dict(
        {
            "scenes": [
                {"id": "s", "tension": 0.1, "variants": v("s"), "next": ["calm", "wild"]},
                {"id": "calm", "tension": low, "variants": v("c"), "next": ["e"]},
                {"id": "wild", "tension": high, "variants": v("w"), "next": ["e"]},
                {"id": "e", "tension": 0.5, "variants": v("e")},
            ],
            "endings": [{"id": "x", "text": "x"}],
        }
    )


def decide(graph, affect):
    return DramaManager(StorySession(graph)).decide("neutral", affect)


def test_curve_interpolates_and_validates():
    c = TensionCurve(((0.0, 0.0), (1.0, 1.0)))
    assert c(0.5) == pytest.approx(0.5) and c(-1) == 0.0 and c(2) == 1.0
    with pytest.raises(ValueError):
        TensionCurve(((0.1, 0.0), (1.0, 1.0)))
    with pytest.raises(ValueError):
        TensionCurve(((0.0, 2.0), (1.0, 1.0)))


def test_viewer_state_classification():
    m = DramaManager(StorySession(fork_graph()))
    assert m.viewer_state(Affect(0.0, 0.0)) == "flat"
    assert m.viewer_state(Affect(0.5, 0.3)) == "engaged"
    assert m.viewer_state(Affect(0.1, 0.9)) == "overwhelmed"


def test_flat_viewer_gets_the_higher_tension_branch():
    # first step: curve target is low, so an engaged viewer gets the calm branch ...
    assert decide(fork_graph(), Affect(0.4, 0.3)).to_id == "calm"
    # ... but a flat (bored) viewer is pushed toward the exciting one
    graph = fork_graph(low=0.2, high=0.6)
    assert decide(graph, Affect(0.0, 0.0)).to_id == "wild"


def test_overwhelmed_viewer_gets_the_calmer_branch():
    graph = fork_graph(low=0.1, high=0.5)  # the curve asks for ~0.29 at step 1
    assert decide(graph, Affect(0.0, 0.95)).to_id == "calm"


def test_decision_has_explainable_reason_and_options():
    d = decide(fork_graph(), Affect(0.0, 0.0))
    assert d.state == "flat" and "aim higher" in d.reason
    assert {o[0] for o in d.options} == {"calm", "wild"}
    assert d.from_id == "s" and d.desired > d.target


def test_terminal_scene_decides_ending():
    s = StorySession(fork_graph())
    s.advance("calm", "neutral", Affect())
    s.advance("e", "neutral", Affect())
    assert DramaManager(s).decide("neutral", Affect()).to_id is None


def test_ties_prefer_the_edge_listed_first():
    g = fork_graph(low=0.5, high=0.5)
    assert decide(g, Affect(0.4, 0.3)).to_id == "calm"


def test_fixed_and_random_baselines():
    g = fork_graph()
    m = DramaManager(StorySession(g))
    assert m.decide_fixed("neutral", Affect(0, 0.9)).to_id == "calm"
    rng = random.Random(1)
    picks = {DramaManager(StorySession(g)).decide_random("neutral", Affect(), rng).to_id
             for _ in range(40)}
    assert picks == {"calm", "wild"}


def test_full_run_on_bundled_story_always_reaches_an_ending():
    g = load_story()
    for aff in [Affect(0, 0), Affect(0.8, 0.9), Affect(-0.6, 0.1), Affect(0.3, 0.3)]:
        s = StorySession(g)
        m = DramaManager(s)
        for _ in range(30):
            d = m.decide("neutral", aff)
            if d.to_id is None:
                s.finish("neutral", aff)
                break
            s.advance(d.to_id, "neutral", aff)
        assert s.finished and s.ending_id in {e.id for e in g.endings}


def test_adaptive_changes_the_route_compared_with_fixed():
    g = load_story()

    def route(decider):
        s = StorySession(g)
        m = DramaManager(s)
        path = [s.current_id]
        while True:
            d = decider(m)
            if d.to_id is None:
                return path
            s.advance(d.to_id, "neutral", Affect(0.0, 0.0))
            path.append(s.current_id)

    flat = route(lambda m: m.decide("neutral", Affect(0.0, 0.0)))
    fixed = route(lambda m: m.decide_fixed("neutral", Affect(0.0, 0.0)))
    assert flat != fixed  # bored viewers are steered away from the default route
