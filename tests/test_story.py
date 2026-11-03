import copy
import json

import pytest

from narrative_engine.config import MOODS
from narrative_engine.director import Story, StoryError, load_story


def test_bundled_story_loads_and_is_complete():
    story = load_story()
    assert len(story.scenes) == 8
    for scene in story.scenes:
        for mood in MOODS:
            assert scene.variants[mood].strip()
    assert story.ending.strip()


def test_advance_walks_all_scenes_then_ending():
    story = load_story()
    first = story.get_line("happy")
    assert first == story.scenes[0].variants["happy"]
    for _ in range(len(story.scenes) - 1):
        assert not story.finished
        story.advance("neutral")
    assert story.advance("neutral") == story.ending
    assert story.finished
    assert story.get_line("sad") == story.ending


def test_reset_returns_to_first_scene():
    story = load_story()
    story.advance("sad")
    story.reset()
    assert story.scene_idx == 0 and not story.finished


def _valid():
    return {
        "title": "t",
        "scenes": [{"id": "a", "variants": {m: f"{m} text" for m in MOODS}}],
        "ending": "the end",
    }


def test_missing_mood_is_rejected():
    data = _valid()
    del data["scenes"][0]["variants"]["sad"]
    with pytest.raises(StoryError):
        Story.from_dict(data)


def test_duplicate_ids_are_rejected():
    data = _valid()
    data["scenes"].append(copy.deepcopy(data["scenes"][0]))
    with pytest.raises(StoryError):
        Story.from_dict(data)


def test_empty_story_and_missing_keys_are_rejected():
    data = _valid()
    data["scenes"] = []
    with pytest.raises(StoryError):
        Story.from_dict(data)
    with pytest.raises(StoryError):
        Story.from_dict({"scenes": []})


def test_bad_file_raises_story_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoryError):
        Story.from_file(bad)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(_valid()), encoding="utf-8")
    assert Story.from_file(ok).title == "t"
