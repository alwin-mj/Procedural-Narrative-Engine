import pytest

from narrative_engine.config import Config, config_from_args


def test_defaults_match_original_prototype():
    c = Config()
    assert (c.width, c.height) == (960, 540)
    assert c.advance_interval_s == 3.5 and c.min_confidence == 0.25
    assert c.motion_threshold == 7.0 and c.happy_bias is True


def test_cli_flags_override_defaults():
    argv = ["--camera", "1", "--interval", "2", "--no-happy-bias", "--backend", "haar"]
    c = config_from_args(argv)
    assert c.camera_index == 1 and c.advance_interval_s == 2 and not c.happy_bias
    assert c.backend == "haar"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "nope"},
        {"smoothing_window": 20, "smoothing_buffer": 10},
        {"min_confidence": 1.5},
        {"advance_interval_s": 0},
        {"width": 100},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
