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


def test_phase5_flags_and_defaults():
    c = Config()
    assert c.threaded and c.perception_hz == 15.0 and c.target_fps == 30.0
    argv = [
        "--no-threads",
        "--perception-hz",
        "10",
        "--perception-width",
        "320",
        "--video",
        "v.avi",
        "--headless",
        "--max-frames",
        "50",
        "--report",
        "r.json",
    ]
    c = config_from_args(argv)
    assert not c.threaded and c.perception_hz == 10 and c.perception_max_width == 320
    assert c.headless and str(c.video) == "v.avi" and c.max_frames == 50


def test_headless_camera_run_needs_a_frame_limit():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        Config(headless=True)
    assert Config(headless=True, max_frames=10).headless
    assert Config(headless=True, video="x.avi").headless


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
        {"log_level": "LOUD"},
        {"camera_index": -1},
        {"smoothing_buffer": 0},
        {"reaction_display_s": -1},
        {"reaction_min_confidence": 2.0},
        {"motion_threshold": -1.0},
        {"hysteresis_margin": -0.1},
        {"mood_dwell_s": -1},
        {"no_face_pause_s": -1},
        {"camera_retry_s": 0},
        {"perception_max_width": -5},
        {"history_len": 0},
        {"perception_hz": 0},
        {"target_fps": -1},
        {"stale_after_s": 3.0, "stale_drop_s": 2.0},
        {"stale_half_life_s": 0},
        {"camera_lost_after": 0},
        {"max_frames": 0},
        {"mood_bias": -0.1},
        {"llm_timeout_s": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
