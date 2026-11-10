"""Central configuration. Every tunable number lives here, not scattered in the code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

MOODS: Tuple[str, ...] = ("happy", "uplift", "sad", "neutral")
BACKENDS: Tuple[str, ...] = ("auto", "mediapipe", "fer", "deepface", "haar")
MOTION_MODES: Tuple[str, ...] = ("face", "frame")
DIRECTOR_MODES: Tuple[str, ...] = ("adaptive", "fixed", "random")
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
    mood_bias: float = 0.0  # extra score for "happy" on neutral/sad readings (0 = off)
    director_mode: str = "adaptive"  # adaptive | fixed | random (the last two are study baselines)
    story_path: Optional[Path] = None  # None = bundled story

    # --- optional LLM scene writer (local Ollama) ---
    llm: Optional[str] = None  # model name, e.g. "llama3.2"; None = authored text only
    llm_host: str = "http://localhost:11434"
    llm_timeout_s: float = 6.0  # per request; the first request gets extra time to load the model

    # --- rendering ---
    history_len: int = 30
    sprites_dir: Optional[Path] = None  # folder of PNG frames; None = glowing orb
    sprite_size: Tuple[int, int] = (160, 160)
    sprite_fps: int = 10

    # --- real-time pipeline ---
    threaded: bool = True  # capture + perception on their own threads (False = one loop)
    target_fps: float = 30.0  # render rate
    perception_hz: float = 15.0  # max detector runs per second
    perception_max_width: int = 640  # frames are shrunk to this width before detection
    stale_after_s: float = 0.5  # result older than this: decay valence/arousal toward neutral
    stale_drop_s: float = 2.0  # result older than this: treat the face as gone
    stale_half_life_s: float = 1.0  # decay speed while stale
    no_face_pause_s: float = 3.0  # no face for this long: pause the story and say so
    camera_lost_after: int = 30  # consecutive failed reads before "camera disconnected"
    camera_retry_s: float = 2.0  # how often to try reopening a lost camera

    # --- headless / replay ---
    video: Optional[Path] = None  # play a video file instead of the webcam
    headless: bool = False  # no window; run to the end of the story / video
    max_frames: Optional[int] = None  # stop after this many frames (required for headless camera)
    report: Optional[Path] = None  # write a JSON run report here (headless)

    # --- evaluation / user study ---
    telemetry: Optional[Path] = None  # directory for JSONL session logs (None = off)
    telemetry_hz: float = 15.0  # max logged readings per second (0 = every reading)
    participant: str = ""  # anonymous id recorded in telemetry and reports
    study: bool = False  # blind UI: hide the director note, stats and detector labels
    seed: Optional[int] = None  # seeds the random baseline mode (reproducible runs)
    replay: Optional[Path] = None  # re-run the director on a recorded telemetry session

    # --- privacy and demo ---
    accept_notice: bool = False  # accept the privacy notice for this run only
    forget_consent: bool = False  # delete the stored consent and exit
    record: Optional[Path] = None  # save the window (with webcam preview) to this video file

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
        if self.director_mode not in DIRECTOR_MODES:
            raise ValueError(f"director_mode must be one of {DIRECTOR_MODES}")
        if self.mood_bias < 0:
            raise ValueError("mood_bias must be >= 0")
        if self.llm_timeout_s <= 0:
            raise ValueError("llm_timeout_s must be > 0")
        if self.target_fps <= 0 or self.perception_hz <= 0:
            raise ValueError("target_fps and perception_hz must be > 0")
        if self.perception_max_width < 0:
            raise ValueError("perception_max_width must be >= 0 (0 = no downscale)")
        if not 0 <= self.stale_after_s < self.stale_drop_s or self.stale_half_life_s <= 0:
            raise ValueError("need 0 <= stale_after_s < stale_drop_s and stale_half_life_s > 0")
        if self.no_face_pause_s < 0 or self.camera_lost_after < 1 or self.camera_retry_s <= 0:
            raise ValueError("invalid no_face_pause_s / camera_lost_after / camera_retry_s")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.headless and self.video is None and self.replay is None and self.max_frames is None:
            raise ValueError("--headless without --video or --replay needs --max-frames")
        if self.record is not None and self.headless:
            raise ValueError("--record saves the window, so it cannot be combined with --headless")
        if self.telemetry_hz < 0:
            raise ValueError("telemetry_hz must be >= 0")
        if self.participant and not all(c.isalnum() or c in "_-" for c in self.participant):
            raise ValueError("participant may only contain letters, digits, '_' and '-'")
        if self.replay is not None and not self.headless:
            raise ValueError("--replay runs without a window; it implies --headless")
        if self.replay is not None and self.video is not None:
            raise ValueError("--replay and --video cannot be combined")
        if self.hysteresis_margin < 0 or self.hysteresis_frames < 1:
            raise ValueError("hysteresis_margin must be >= 0 and hysteresis_frames >= 1")
        if not 0.0 < self.affect_smoothing <= 1.0:
            raise ValueError("affect_smoothing must be in (0, 1]")
        if self.calibration_s <= 0 or self.mood_dwell_s < 0:
            raise ValueError("calibration_s must be > 0 and mood_dwell_s >= 0")
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")

    @property
    def resolved_motion_threshold(self) -> float:
        """The explicit threshold, or the default for the chosen motion mode."""
        if self.motion_threshold is not None:
            return self.motion_threshold
        return DEFAULT_MOTION_THRESHOLD[self.motion_mode]


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
        "--mood-bias",
        type=float,
        default=d.mood_bias,
        help="extra 'happy' score on neutral/sad readings (0 = off, 0.25 = old prototype)",
    )
    p.add_argument(
        "--mode",
        choices=DIRECTOR_MODES,
        default=d.director_mode,
        help="adaptive (default), or the fixed / random baselines used for evaluation",
    )
    p.add_argument(
        "--llm", default=None, metavar="MODEL", help="rewrite scenes with a local Ollama model"
    )
    p.add_argument("--llm-host", default=d.llm_host, help="Ollama server URL")
    p.add_argument(
        "--llm-timeout", type=float, default=d.llm_timeout_s, help="seconds to wait per LLM request"
    )
    p.add_argument(
        "--no-threads",
        action="store_true",
        help="run capture, detection and rendering in one loop (simpler to debug)",
    )
    p.add_argument("--fps", type=float, default=d.target_fps, help="render frame rate")
    p.add_argument(
        "--perception-hz", type=float, default=d.perception_hz, help="max detector runs per second"
    )
    p.add_argument(
        "--perception-width",
        type=int,
        default=d.perception_max_width,
        help="shrink frames to this width before detection (0 = off)",
    )
    p.add_argument("--video", type=Path, default=None, help="play a video file, not the webcam")
    p.add_argument("--headless", action="store_true", help="no window; run until the end")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    p.add_argument("--report", type=Path, default=None, help="write a JSON run report (headless)")
    p.add_argument(
        "--telemetry",
        nargs="?",
        const=Path("logs"),
        default=None,
        type=Path,
        metavar="DIR",
        help="write a JSONL session log to DIR (default: logs/). No video is ever recorded",
    )
    p.add_argument(
        "--telemetry-hz", type=float, default=d.telemetry_hz, help="max logged readings per second"
    )
    p.add_argument("--participant", default="", metavar="ID", help="anonymous participant id")
    p.add_argument("--study", action="store_true", help="blind UI for user studies")
    p.add_argument("--seed", type=int, default=None, help="seed for --mode random")
    p.add_argument(
        "--replay",
        type=Path,
        default=None,
        metavar="FILE",
        help="re-run the director on a telemetry log (no camera, implies --headless)",
    )
    p.add_argument(
        "--yes",
        "--accept-privacy-notice",
        dest="accept_notice",
        action="store_true",
        help="accept the privacy notice for this run (nothing is remembered)",
    )
    p.add_argument(
        "--forget-consent",
        action="store_true",
        help="delete the remembered privacy consent and exit",
    )
    p.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="FILE",
        help="save the window, including your webcam preview, to FILE (.mp4 or .avi)",
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
        mood_bias=a.mood_bias,
        director_mode=a.mode,
        accept_notice=a.accept_notice,
        forget_consent=a.forget_consent,
        record=a.record,
        telemetry=a.telemetry,
        telemetry_hz=a.telemetry_hz,
        participant=a.participant,
        study=a.study,
        seed=a.seed,
        replay=a.replay,
        threaded=not a.no_threads,
        target_fps=a.fps,
        perception_hz=a.perception_hz,
        perception_max_width=a.perception_width,
        video=a.video,
        headless=a.headless or a.replay is not None,
        max_frames=a.max_frames,
        report=a.report,
        llm=a.llm,
        llm_host=a.llm_host,
        llm_timeout_s=a.llm_timeout,
        log_level=a.log_level,
    )
