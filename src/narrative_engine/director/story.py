"""Story data model, loaded from JSON (story text is data, not code).

JSON format::

    {
      "title": "Arin's Day",
      "scenes": [
        {"id": "wake", "variants": {"neutral": "...", "happy": "...", "uplift": "...",
                                    "sad": "..."}}
      ],
      "ending": "..."
    }

Every scene must define a non-empty line for each mood in ``config.MOODS``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from ..config import MOODS

DEFAULT_STORY = Path(__file__).resolve().parent.parent / "data" / "stories" / "arin_adventure.json"


class StoryError(ValueError):
    """Raised when story data is malformed."""


@dataclass(frozen=True)
class Scene:
    id: str
    variants: Mapping[str, str]

    def line(self, mood: str) -> str:
        return self.variants.get(mood) or self.variants["neutral"]


class Story:
    """A linear sequence of scenes plus an ending. (Branching arrives in a later phase.)"""

    def __init__(self, scenes: List[Scene], ending: str, title: str = "") -> None:
        self.scenes = scenes
        self.ending = ending
        self.title = title
        self.scene_idx = 0
        self._validate()

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Story":
        try:
            raw_scenes = data["scenes"]
            ending = data["ending"]
        except (KeyError, TypeError) as exc:
            raise StoryError(f"story must contain 'scenes' and 'ending': {exc}") from exc
        scenes = []
        for i, raw in enumerate(raw_scenes):
            if not isinstance(raw, Mapping) or "variants" not in raw:
                raise StoryError(f"scene #{i} must be an object with 'variants'")
            scene_id = str(raw.get("id", f"scene_{i}"))
            scenes.append(Scene(id=scene_id, variants=dict(raw["variants"])))
        return cls(scenes, str(ending), str(data.get("title", "")))

    @classmethod
    def from_file(cls, path: Path) -> "Story":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise StoryError(f"cannot read story file {path}: {exc}") from exc
        return cls.from_dict(data)

    def _validate(self) -> None:
        if not self.scenes:
            raise StoryError("story needs at least one scene")
        if not self.ending.strip():
            raise StoryError("story needs a non-empty ending")
        seen = set()
        for sc in self.scenes:
            if sc.id in seen:
                raise StoryError(f"duplicate scene id: {sc.id!r}")
            seen.add(sc.id)
            for mood in MOODS:
                text = sc.variants.get(mood)
                if not isinstance(text, str) or not text.strip():
                    raise StoryError(f"scene {sc.id!r} is missing a line for mood {mood!r}")

    # ---------------------------------------------------------------- playback
    @property
    def finished(self) -> bool:
        return self.scene_idx >= len(self.scenes)

    def reset(self) -> None:
        self.scene_idx = 0

    def get_line(self, mood: str) -> str:
        """Current scene's line for ``mood`` without advancing."""
        if self.finished:
            return self.ending
        return self.scenes[self.scene_idx].line(mood)

    def advance(self, mood: str) -> str:
        """Move to the next scene and return its line; after the last scene return the ending."""
        if self.scene_idx < len(self.scenes) - 1:
            self.scene_idx += 1
            return self.get_line(mood)
        self.scene_idx = len(self.scenes)
        return self.ending


def load_story(path: Optional[Path] = None) -> Story:
    """Load ``path`` or the bundled default story."""
    return Story.from_file(path or DEFAULT_STORY)
