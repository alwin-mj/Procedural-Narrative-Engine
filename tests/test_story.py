import copy
import json

import pytest

from narrative_engine.config import MOODS
from narrative_engine.director import StoryError, StoryGraph, load_story
from narrative_engine.director.graph import Context, parse_condition


def variants(tag="x"):
    return {m: f"{tag} {m} text" for m in MOODS}


def valid():
    return {
        "title": "t",
        "scenes": [
            {"id": "a", "variants": variants("a"), "next": [{"to": "b"}]},
            {"id": "b", "variants": variants("b")},
        ],
        "endings": [{"id": "end", "text": "the end"}],
    }


def test_bundled_story_is_a_valid_branching_graph():
    g = load_story()
    assert len(g.nodes) >= 8
    assert g.shortest_path() < g.longest_path()  # real branching
    assert len(g.endings) >= 3 and g.endings[-1].when.always
    assert all(0.0 <= n.tension <= 1.0 for n in g.nodes.values())
    assert any(not e.when.always for n in g.nodes.values() for e in n.edges)


def test_every_route_through_bundled_story_reaches_an_ending():
    g = load_story()

    def walk(nid, depth=0):
        assert depth < 50
        n = g.nodes[nid]
        if n.terminal:
            return
        for e in n.edges:
            walk(e.to, depth + 1)

    walk(g.start)


def test_legacy_linear_format_still_loads_as_a_chain():
    data = {"scenes": [{"id": s, "variants": variants(s)} for s in "abc"], "ending": "bye"}
    g = StoryGraph.from_dict(data)
    assert [e.to for e in g.nodes["a"].edges] == ["b"]
    assert g.nodes["c"].terminal and g.longest_path() == g.shortest_path() == 3
    assert g.endings[0].text == "bye"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["scenes"][0]["variants"].pop("sad"),  # missing mood line
        lambda d: d["scenes"].append(copy.deepcopy(d["scenes"][1])),  # duplicate id
        lambda d: d["scenes"][0]["next"].append({"to": "ghost"}),  # unknown target
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"mood": "sad"}}]),  # no default
        lambda d: d["scenes"][1].update(next=[{"to": "a"}]),  # cycle
        lambda d: d["scenes"].append({"id": "c", "variants": variants("c")}),  # unreachable
        lambda d: d["scenes"][0].update(tension=1.5),  # tension out of range
        lambda d: d["endings"][0].update(when={"avg_valence": ">0"}),  # no default ending
        lambda d: d.update(endings=[]),  # no endings
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"nonsense": 1}}]),
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"valence": "big"}}]),
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"mood": "angry"}}]),
        lambda d: d.update(start="zzz"),
    ],
)
def test_invalid_stories_are_rejected(mutate):
    data = valid()
    mutate(data)
    with pytest.raises(StoryError):
        StoryGraph.from_dict(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(scenes="not a list"),
        lambda d: d["scenes"].__setitem__(0, "not an object"),
        lambda d: d["scenes"][0].update(tension="high"),
        lambda d: d["scenes"][0].update(next=[{"when": {"valence": ">0"}}]),  # edge without 'to'
        lambda d: d["scenes"][0].update(next=[{"to": "a"}, {"to": "b"}]),  # edge to itself
        lambda d: d["scenes"][0].update(effects=["bad"]),
        lambda d: d["scenes"][0].update(effects=[{"set_flags": "oops"}]),
        lambda d: d["scenes"][0].update(effects=[{"relations": {"bob": "lots"}}]),
        lambda d: d["scenes"][0].update(callbacks=[{"if": {}}]),  # no text
        lambda d: d["scenes"][0].update(callbacks=[{"text": "  "}]),  # blank text
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": "oops"}]),
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"relation": "x"}}]),
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"flag": []}}]),
        lambda d: d["scenes"][0].update(next=[{"to": "b", "when": {"flag": [1]}}]),
        lambda d: d["endings"].insert(0, {"id": "end", "text": "dup", "when": {"mood": "sad"}}),
        lambda d: d["endings"][0].update(text=" "),
        lambda d: d["endings"].__setitem__(0, {"id": "x"}),  # ending without text
    ],
)
def test_more_malformed_stories_are_rejected_with_story_error(mutate):
    data = valid()
    mutate(data)
    with pytest.raises(StoryError):
        StoryGraph.from_dict(data)


def test_non_object_story_and_legacy_bad_scene_are_rejected():
    with pytest.raises(StoryError):
        StoryGraph.from_dict([])
    with pytest.raises(StoryError):
        StoryGraph.from_dict({"scenes": [{"id": "a"}], "ending": "x"})  # no variants


def test_path_lengths_and_terminals():
    g = StoryGraph.from_dict(valid())
    assert g.longest_path() == g.shortest_path() == 2 and g.terminals == ["b"]
    assert g.longest_path("b") == 1


def test_valid_minimal_story_loads():
    assert StoryGraph.from_dict(valid()).start == "a"


def test_bad_files_raise_story_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoryError):
        StoryGraph.from_file(bad)
    with pytest.raises(StoryError):
        StoryGraph.from_file(tmp_path / "missing.json")
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(valid()), encoding="utf-8")
    assert StoryGraph.from_file(ok).title == "t"


# ---------------------------------------------------------------- conditions
def test_numeric_conditions_and_ranges():
    c = parse_condition({"valence": [">-0.2", "<0.4"], "arousal": ">=0.5"}, "t")
    assert c.test(Context(valence=0.1, arousal=0.5))
    assert not c.test(Context(valence=0.5, arousal=0.9))
    assert not c.test(Context(valence=0.1, arousal=0.4))


def test_flag_visited_mood_and_relation_conditions():
    ctx = Context(
        mood="happy", flags=frozenset({"pendant"}), visited=("a", "b"), relations={"smith": 2}
    )
    assert parse_condition({"flag": "pendant"}, "t").test(ctx)
    assert not parse_condition({"not_flag": "pendant"}, "t").test(ctx)
    assert parse_condition({"visited": ["a", "b"]}, "t").test(ctx)
    assert parse_condition({"not_visited": "z"}, "t").test(ctx)
    assert parse_condition({"mood": ["happy", "sad"]}, "t").test(ctx)
    assert parse_condition({"relation": {"smith": ">=2"}}, "t").test(ctx)
    assert not parse_condition({"relation": {"baker": ">=1"}}, "t").test(ctx)


def test_empty_condition_is_always_true():
    assert parse_condition(None, "t").always
    assert parse_condition({}, "t").test(Context())
