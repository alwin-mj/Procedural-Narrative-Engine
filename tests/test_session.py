from narrative_engine.config import MOODS
from narrative_engine.director import Affect, StoryError, StoryGraph, StorySession, load_story


def v(tag):
    return {m: f"{tag}-{m}" for m in MOODS}


def small_graph():
    return StoryGraph.from_dict(
        {
            "start": "a",
            "protagonist": "Z",
            "scenes": [
                {
                    "id": "a",
                    "variants": v("a"),
                    "effects": [
                        {"when": {"mood": "happy"}, "set_flags": ["gift"], "relations": {"bob": 1}}
                    ],
                    "next": [{"to": "b"}, {"to": "c", "when": {"arousal": ">=0.5"}}],
                },
                {
                    "id": "b",
                    "variants": v("b"),
                    "callbacks": [{"if": {"flag": "gift"}, "text": "(gift!)"}],
                },
                {"id": "c", "variants": v("c")},
            ],
            "endings": [
                {"id": "joy", "when": {"avg_valence": ">=0.3"}, "text": "joy"},
                {"id": "calm", "text": "calm"},
            ],
        }
    )


def test_session_starts_at_start_and_reports_line():
    s = StorySession(small_graph())
    assert s.current_id == "a" and s.step_count == 0
    assert s.get_line("sad") == "a-sad"


def test_candidates_respect_conditions():
    s = StorySession(small_graph())
    assert [e.to for e in s.candidates("neutral", Affect(0, 0.1))] == ["b"]
    assert [e.to for e in s.candidates("neutral", Affect(0, 0.9))] == ["b", "c"]


def test_effects_set_flags_and_callbacks_use_them():
    s = StorySession(small_graph())
    s.advance("b", "neutral", Affect())
    assert "gift" not in s.state.flags  # effect of 'a' only fires if a was entered happy
    assert s.get_line("neutral") == "b-neutral"

    s2 = StorySession(small_graph())
    s2.state.flags.add("gift")  # as if earned earlier
    s2.advance("b", "happy", Affect())
    assert s2.get_line("happy") == "b-happy (gift!)"


def test_effects_fire_on_entering_with_matching_mood():
    g = StoryGraph.from_dict(
        {
            "scenes": [
                {"id": "a", "variants": v("a"), "next": ["b"]},
                {
                    "id": "b",
                    "variants": v("b"),
                    "effects": [
                        {"when": {"mood": "happy"}, "set_flags": ["f"], "relations": {"bob": 2}}
                    ],
                },
            ],
            "endings": [{"id": "e", "text": "e"}],
        }
    )
    s = StorySession(g)
    s.advance("b", "happy", Affect())
    assert "f" in s.state.flags and s.state.relations["bob"] == 2
    s.reset()
    s.advance("b", "sad", Affect())
    assert not s.state.flags


def test_invalid_moves_are_rejected():
    s = StorySession(small_graph())
    try:
        s.advance("zzz", "neutral", Affect())
        raise AssertionError("expected StoryError")
    except StoryError:
        pass
    try:
        s.finish("neutral", Affect())  # 'a' is not terminal
        raise AssertionError("expected StoryError")
    except StoryError:
        pass


def test_ending_depends_on_whole_trajectory():
    g = small_graph()
    s = StorySession(g)
    s.advance("b", "neutral", Affect(0.6, 0.1))
    assert s.finish("neutral", Affect(0.5, 0.1)) == "joy" and s.ending_id == "joy"

    s2 = StorySession(g)
    s2.advance("b", "neutral", Affect(-0.2, 0.1))
    assert s2.finish("neutral", Affect(0.0, 0.1)) == "calm"
    assert s2.finished and s2.get_line("happy") == "calm"


def test_preview_line_matches_real_line_and_does_not_mutate():
    s = StorySession(small_graph())
    before = (s.current_id, set(s.state.flags), list(s.state.visited))
    preview = s.preview_line("b", "happy", Affect(0.4, 0.2))
    assert (s.current_id, set(s.state.flags), list(s.state.visited)) == before
    s.advance("b", "happy", Affect(0.4, 0.2))
    assert s.get_line("happy") == preview


def test_reset_restarts_cleanly():
    s = StorySession(load_story())
    s.advance(s.node.edges[0].to, "happy", Affect(0.3, 0.1))
    s.reset()
    assert s.step_count == 0 and s.state.n_scenes == 0 and not s.finished
