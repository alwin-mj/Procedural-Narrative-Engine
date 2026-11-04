import pytest

from narrative_engine.config import Config, config_from_args


def test_defaults_match_original_prototype():
    c = Config()
    assert (c.width, c.height) == (960, 540)
    assert c.advance_interval_s == 3.5 and c.min_confidence == 0.25
    assert c.mood_bias == 0.0 and c.director_mode == "adaptive" and c.llm is None
    assert c.motion_mode == "face" and c.resolved_motion_threshold == 0.03


def test_cli_flags_override_defaults():
    argv = ["--camera", "1", "--interval", "2", "--mood-bias", "0.25", "--backend", "haar"]
    c = config_from_args(argv)
    assert c.camera_index == 1 and c.advance_interval_s == 2 and c.mood_bias == 0.25
    assert c.backend == "haar"


def test_motion_mode_changes_default_threshold():
    c = config_from_args(["--motion-mode", "frame"])
    assert c.resolved_motion_threshold == 7.0
    assert config_from_args(["--motion-threshold", "0.1"]).resolved_motion_threshold == 0.1


def test_phase3_flags():
    c = config_from_args(["--no-calibration", "--mp-model", "m.task", "--backend", "mediapipe"])
    assert not c.calibrate_on_start and str(c.mediapipe_model) == "m.task"
    assert c.backend == "mediapipe"


def test_phase4_flags():
    c = config_from_args(["--mode", "fixed", "--llm", "llama3.2", "--llm-timeout", "3"])
    assert c.director_mode == "fixed" and c.llm == "llama3.2" and c.llm_timeout_s == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "nope"},
        {"smoothing_window": 20, "smoothing_buffer": 10},
        {"min_confidence": 1.5},
        {"advance_interval_s": 0},
        {"width": 100},
        {"motion_mode": "nope"},
        {"hysteresis_frames": 0},
        {"affect_smoothing": 0.0},
        {"calibration_s": 0},
        {"director_mode": "chaos"},
        {"mood_bias": -0.1},
        {"llm_timeout_s": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
