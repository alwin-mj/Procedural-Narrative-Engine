"""Central configuration. Every tunable number lives here, not scattered in the code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

MOODS: Tuple[str, ...] = ("happy", "uplift", "sad", "neutral")
BACKENDS: Tuple[str, ...] = ("auto", "mediapipe", "fer", "deepface", "haar")
MOTION_MODES: Tuple[str, ...] = ("face", "frame")
DEFAULT_MOTION_THRESHOLD = {"face": 0.03, "frame": 7.0}
LOG_LEVELS: Tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class Config:
    """Runtime settings with defaults identical to the original prototype."""

    # --- input / window ---
    camera_index: int = 0
    width: int = 960
    height: int = 540
    window_title: str = "Narrative Engine"

    # --- perception ---
    backend: str = "auto"  # auto = MediaPipe -> FER -> DeepFace -> Haar, first available wins
    mediapipe_model: Optional[Path] = None  # None = download/cache face_landmarker.task
    smoothing_buffer: int = 10  # frames kept in the rolling buffer
    smoothing_window: int = 6  # latest frames averaged for the stable mood
    hysteresis_margin: float = 0.08  # new mood must beat the held mood by this much...
    hysteresis_frames: int = 3  # ...for this many consecutive frames before switching
    affect_smoothing: float = 0.3  # EMA factor for valence/arousal (1 = no smoothing)
    calibrate_on_start: bool = True  # neutral-face calibration the first time you press S
    calibration_s: float = 3.0  # how long to hold a neutral face
    motion_mode: str = "face"  # "face" = face-box movement, "frame" = whole-frame difference
    motion_threshold: Optional[float] = None  # None = default for the chosen motion_mode

    # --- director ---
    advance_interval_s: float = 3.5  # seconds between automatic scene changes
    min_confidence: float = 0.25  # minimum stable confidence to auto-advance
    reaction_min_confidence: float = 0.20  # minimum confidence to show a mood-shift line
    reaction_display_s: float = 2.0  # how long the "A shift: ..." prefix stays visible
    mood_dwell_s: float = 0.8  # a mood must hold this long before it counts as a "shift"
    happy_bias: bool = True  # gently steer neutral/sad moods toward warmer scenes
    story_path: Optional[Path] = None  # None = bundled story

    # --- rendering ---
    history_len: int = 30
    sprites_dir: Optional[Path] = None  # folder of PNG frames; None = glowing orb
    sprite_size: Tuple[int, int] = (160, 160)
    sprite_fps: int = 10

    # --- misc ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.log_level.upper() not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {LOG_LEVELS}, got {self.log_level!r}")
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        if self.width < 320 or self.height < 240:
            raise ValueError("window must be at least 320x240")
        if self.smoothing_buffer < 1 or self.smoothing_window < 1:
            raise ValueError("smoothing_buffer and smoothing_window must be >= 1")
        if self.smoothing_window > self.smoothing_buffer:
            raise ValueError("smoothing_window cannot exceed smoothing_buffer")
        if self.advance_interval_s <= 0 or self.reaction_display_s < 0:
            raise ValueError("advance_interval_s must be > 0 and reaction_display_s >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.reaction_min_confidence <= 1.0:
            raise ValueError("reaction_min_confidence must be in [0, 1]")
        if self.motion_mode not in MOTION_MODES:
            raise ValueError(f"motion_mode must be one of {MOTION_MODES}, got {self.motion_mode!r}")
        if self.motion_threshold is not None and self.motion_threshold < 0:
            raise ValueError("motion_threshold must be >= 0")
        if self.hysteresis_margin < 0 or self.hysteresis_frames < 1:
            raise ValueError("hysteresis_margin must be >= 0 and hysteresis_frames >= 1")
        if not 0.0 < self.affect_smoothing <= 1.0:
            raise ValueError("affect_smoothing must be in (0, 1]")
        if self.calibration_s <= 0 or self.mood_dwell_s < 0:
            raise ValueError("calibration_s must be > 0 and mood_dwell_s >= 0")

    @property
    def resolved_motion_threshold(self) -> float:
        """The explicit threshold, or the default for the chosen motion mode."""
        if self.motion_threshold is not None:
            return self.motion_threshold
        return DEFAULT_MOTION_THRESHOLD[self.motion_mode]
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="narrative-engine",
        description="Webcam-driven interactive story that adapts to your expression.",
    )
    p.add_argument("--camera", type=int, default=d.camera_index, help="camera index (default: 0)")
    p.add_argument("--width", type=int, default=d.width, help="window width")
    p.add_argument("--height", type=int, default=d.height, help="window height")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=d.backend,
        help="emotion backend (default: auto = MediaPipe -> FER -> DeepFace -> Haar)",
    )
    p.add_argument(
        "--mp-model",
        type=Path,
        default=None,
        help="path to face_landmarker.task (skips the one-time download)",
    )
    p.add_argument("--story", type=Path, default=None, help="path to a story JSON file")
    p.add_argument("--sprites", type=Path, default=None, help="folder of PNG frames for the player")
    p.add_argument(
        "--interval", type=float, default=d.advance_interval_s, help="seconds between scenes"
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=d.min_confidence,
        help="confidence needed to advance",
    )
    p.add_argument(
        "--motion-mode",
        choices=MOTION_MODES,
        default=d.motion_mode,
        help="walking detection: face-box movement (default) or whole-frame difference",
    )
    p.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="walking sensitivity (default depends on --motion-mode)",
    )
    p.add_argument(
        "--no-calibration",
        action="store_true",
        help="skip the neutral-face calibration on start",
    )
    p.add_argument(
        "--no-happy-bias",
        action="store_true",
        help="disable the gentle steering of neutral/sad moods toward warmer scenes",
    )
    p.add_argument("--log-level", choices=LOG_LEVELS, default=d.log_level, help="log verbosity")
    return p


def config_from_args(argv: Optional[Sequence[str]] = None) -> Config:
    a = build_parser().parse_args(argv)
    return Config(
        camera_index=a.camera,
        width=a.width,
        height=a.height,
        backend=a.backend,
        mediapipe_model=a.mp_model,
        story_path=a.story,
        sprites_dir=a.sprites,
        advance_interval_s=a.interval,
        min_confidence=a.min_confidence,
        motion_mode=a.motion_mode,
        motion_threshold=a.motion_threshold,
        calibrate_on_start=not a.no_calibration,
        happy_bias=not a.no_happy_bias,
        log_level=a.log_level,
    )
