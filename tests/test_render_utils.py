import numpy as np

from narrative_engine.perception import MotionDetector
from narrative_engine.render import GameWindow, wrap_text


def test_wrap_text_respects_width_and_keeps_all_words():
    text = "one two three four five six seven eight nine ten"
    lines = wrap_text(text, 12)
    assert all(len(line) <= 12 for line in lines)
    assert " ".join(lines) == text


def test_wrap_text_handles_empty_and_long_words():
    assert wrap_text("", 10) == [""]
    assert wrap_text("supercalifragilistic", 5) == ["supercalifragilistic"]


def test_motion_detector_flags_big_changes_only():
    m = MotionDetector(threshold=7.0)
    still = np.full((60, 80, 3), 100, dtype=np.uint8)
    assert m.update(still) is False  # first frame: nothing to compare
    assert m.update(still.copy()) is False
    assert m.update(np.full((60, 80, 3), 200, dtype=np.uint8)) is True


def test_window_renders_expected_shape_for_every_mood():
    win = GameWindow(320, 240)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for mood in ("happy", "uplift", "sad", "neutral", "unknown"):
        out = win.render(frame, mood, 0.5, (10, 10, 40, 40), True, "hello world", ["happy"] * 5)
        assert out.shape == (240, 320, 3) and out.dtype == np.uint8
    out = win.render(
        None,
        "neutral",
        0.0,
        None,
        False,
        "",
        None,
        backend="haar",
        stats="30 fps | detector 15 Hz | p95 12 ms",
    )
    assert out.shape == (240, 320, 3)
