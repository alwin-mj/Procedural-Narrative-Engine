"""Every bundled story, the story linter, and name-based story selection."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import narrative_engine.__main__ as entry
from narrative_engine import story_tool
from narrative_engine.app import App
from narrative_engine.config import Config
from narrative_engine.director import (
    Affect,
    DramaManager,
    StoryError,
    StoryGraph,
    StorySession,
    bundled_stories,
    load_story,
    resolve_story,
    story_warnings,
)
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import VideoFileSource

NAMES = sorted(bundled_stories())
# constant viewers that between them should reach every ending of a well-built story
PROFILES = [Affect(0.5, 0.1), Affect(0.0, 0.5), Affect(-0.3, 0.1), Affect(0.0, 0.1)]


def play(graph, affect, mood="neutral"):
    session = StorySession(graph)
    manager = DramaManager(session)
    for _ in range(50):
        decision = manager.decide(mood, affect)
        if decision.to_id is None:
            session.finish(mood, affect)
            return session
        session.advance(decision.to_id, mood, affect)
    raise AssertionError("the story never ended")


def test_the_expected_stories_ship_with_the_package():
    assert {"arin_adventure", "last_train", "starter"} <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_every_bundled_story_is_valid_and_lint_clean(name):
    graph = load_story(Path(name))
    assert graph.title and graph.protagonist and len(graph.nodes) >= 3
    assert story_warnings(graph) == []
    assert graph.shortest_path() <= graph.longest_path()
    for node in graph.nodes.values():
        assert all(len(text) >= 30 for text in node.variants.values()), node.id


@pytest.mark.parametrize("name", NAMES)
def test_every_ending_of_every_story_can_actually_be_reached(name):
    graph = load_story(Path(name))
    reached = {play(graph, affect).ending_id for affect in PROFILES}
    assert reached == {e.id for e in graph.endings}, f"unreachable endings in {name}"


@pytest.mark.parametrize("name", NAMES)
def test_stories_react_to_the_viewer_not_just_to_time(name):
    graph = load_story(Path(name))
    happy, flat = play(graph, Affect(0.6, 0.1), "happy"), play(graph, Affect(0.0, 0.0))
    assert happy.state.moods != flat.state.moods or happy.state.visited != flat.state.visited


def test_the_second_story_is_a_different_shape_not_a_reskin():
    arin, train = load_story(Path("arin_adventure")), load_story(Path("last_train"))
    assert set(arin.nodes).isdisjoint(train.nodes)
    assert (len(arin.nodes), arin.longest_path()) != (len(train.nodes), train.longest_path())
    assert {e.id for e in arin.endings}.isdisjoint({e.id for e in train.endings})


# ---------------------------------------------------------------- selecting a story by name
def test_a_bare_name_selects_a_bundled_story_and_a_real_file_always_wins(tmp_path):
    assert resolve_story(Path("last_train")).name == "last_train.json"
    assert resolve_story(Path("last_train.json")).name == "last_train.json"
    assert resolve_story(None).name == "arin_adventure.json"
    mine = tmp_path / "last_train.json"  # same name as a bundled story, but a real file
    mine.write_text("{}", encoding="utf-8")
    assert resolve_story(mine) == mine


def test_unknown_names_list_what_is_available():
    with pytest.raises(StoryError, match="last_train"):
        resolve_story(Path("nope"))
    with pytest.raises(StoryError, match="story not found"):
        resolve_story(Path("somewhere/last_train"))  # a path that does not exist is not a name


def test_load_story_accepts_names():
    assert load_story(Path("starter")).title == "Starter Story"


# ---------------------------------------------------------------- the linter
def moods(text):
    return {m: f"{text} {m} line with enough words" for m in ("neutral", "happy", "uplift", "sad")}


def graph(scenes, endings=None):
    endings = endings or [{"id": "e", "text": "the end"}]
    return StoryGraph.from_dict({"scenes": scenes, "endings": endings})


def test_a_clean_story_has_no_warnings():
    g = graph(
        [
            {
                "id": "a",
                "variants": moods("a"),
                "next": ["b"],
                "effects": [
                    {"when": {"mood": "happy"}, "set_flags": ["f"], "relations": {"bob": 1}}
                ],
            },
            {
                "id": "b",
                "variants": moods("b"),
                "callbacks": [{"if": {"flag": "f", "relation": {"bob": ">=1"}}, "text": "later"}],
            },
        ]
    )
    assert story_warnings(g) == []


def test_the_linter_finds_dead_state_and_copy_paste_mistakes():
    same = {m: "exactly the same sentence for every mood" for m in moods("x")}
    g = graph(
        [
            {
                "id": "a",
                "variants": moods("a"),
                "next": ["b"],
                "effects": [{"set_flags": ["unused_flag"], "relations": {"unused_rel": 1}}],
            },
            {
                "id": "b",
                "variants": same,
                "callbacks": [
                    {
                        "if": {
                            "flag": "never_set",
                            "relation": {"never_changed": ">=1"},
                            "visited": "ghost",
                        },
                        "text": "x",
                    }
                ],
            },
        ]
    )
    text = "\n".join(story_warnings(g))
    for expected in (
        "'never_set' is checked",
        "'unused_flag' is set",
        "'unused_rel' is changed",
        "'never_changed' is checked",
        "unknown scene 'ghost'",
        "identical",
    ):
        assert expected in text, expected
    assert story_warnings(g) == sorted(story_warnings(g))


def test_flags_used_only_in_edges_or_endings_count_as_used():
    g = graph(
        [
            {
                "id": "a",
                "variants": moods("a"),
                "next": [{"to": "b", "when": {"flag": "k"}}, "c"],
                "effects": [{"set_flags": ["k", "j"]}],
            },
            {"id": "b", "variants": moods("b")},
            {"id": "c", "variants": moods("c")},
        ],
        [{"id": "x", "when": {"not_flag": "j"}, "text": "x"}, {"id": "y", "text": "y"}],
    )
    assert story_warnings(g) == []


# ---------------------------------------------------------------- the command-line tool
def test_story_tool_list_check_and_strict(tmp_path, capsys):
    assert story_tool.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "last_train" in out and "starter" in out and "scenes" in out
    assert story_tool.main(["check", "last_train"]) == 0
    assert "OK: The Last Train" in capsys.readouterr().out
    assert story_tool.main(["check", "--all", "--strict"]) == 0

    flawed = tmp_path / "flawed.json"
    data = json.loads(Path(resolve_story(Path("starter"))).read_text(encoding="utf-8"))
    data["scenes"][0]["effects"] = [{"set_flags": ["orphan"]}]
    flawed.write_text(json.dumps(data), encoding="utf-8")
    assert story_tool.main(["check", str(flawed)]) == 0  # legal, but...
    assert "warning:" in capsys.readouterr().out
    assert story_tool.main(["check", str(flawed), "--strict"]) == 1  # ...strict mode objects
    assert story_tool.main(["check", "does_not_exist"]) == 1
    assert "INVALID" in capsys.readouterr().err
    assert story_tool.main(["mermaid", "starter"]) == 0
    assert "flowchart TD" in capsys.readouterr().out


# ---------------------------------------------------------------- running the engine on a pack
class Viewer:
    supports_calibration = False
    calibrated = False
    active_backend = "fake"

    def detect_stable(self, frame):
        scores = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
        return Detection("happy", 0.9, (50, 40, 80, 80), scores, 0.6, 0.3)


def write_video(path, frames=400, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for _ in range(frames):
        w.write(np.full((120, 160, 3), 60, dtype=np.uint8))
    w.release()


def test_the_engine_plays_a_story_pack_selected_by_name(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video)
    cfg = Config(
        backend="haar",
        video=video,
        headless=True,
        calibrate_on_start=False,
        advance_interval_s=1.0,
        story_path=Path("last_train"),
    )
    report = App(cfg, detector=Viewer()).run_headless(VideoFileSource(video))
    assert report["route"][0] == "platform" and report["route"][-1] == "train"
    assert report["finished"] and report["ending"] in {
        "warm_departure",
        "adrenaline",
        "quiet_ride",
        "late_night",
    }


def test_the_command_line_accepts_story_names_and_rejects_unknown_ones(tmp_path, capsys):
    video, out = tmp_path / "v.avi", tmp_path / "r.json"
    write_video(video, frames=30)
    base = ["--backend", "haar", "--video", str(video), "--headless", "--no-calibration"]
    assert entry.main(base + ["--story", "starter", "--report", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["route"][0] == "intro"
    assert entry.main(base + ["--story", "nope"]) == 2
    err = capsys.readouterr().err
    assert "story not found" in err and "last_train" in err
