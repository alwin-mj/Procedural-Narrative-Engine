"""Director: story graph, playthrough state, and the drama manager that steers it."""

from .drama import DEFAULT_CURVE, Decision, DramaConfig, DramaManager, TensionCurve
from .graph import (
    Condition,
    Context,
    StoryError,
    StoryGraph,
    bundled_stories,
    load_story,
    resolve_story,
    story_warnings,
)
from .selector import choose_advance_emotion
from .session import Affect, StorySession, StoryState

__all__ = [
    "Affect",
    "Condition",
    "Context",
    "DEFAULT_CURVE",
    "Decision",
    "DramaConfig",
    "DramaManager",
    "StoryError",
    "StoryGraph",
    "StorySession",
    "StoryState",
    "TensionCurve",
    "bundled_stories",
    "choose_advance_emotion",
    "load_story",
    "resolve_story",
    "story_warnings",
]
