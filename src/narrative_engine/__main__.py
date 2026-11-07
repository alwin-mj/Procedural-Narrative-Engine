"""Entry point: ``python -m narrative_engine`` or the ``narrative-engine`` command."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .app import App
from .config import Config, config_from_args
from .director import StoryError


def missing_input(cfg: Config) -> Optional[str]:
    """A message if a file the user named does not exist, else ``None``.

    Checked before anything heavy is loaded, so a typo fails in a blink instead of after the
    face model has started up.
    """
    for flag, path in (
        ("--video", cfg.video),
        ("--replay", cfg.replay),
        ("--story", cfg.story_path),
        ("--mp-model", cfg.mediapipe_model),
    ):
        if path is not None and not Path(path).is_file():
            return f"{flag}: file not found: {path}"
    if cfg.sprites_dir is not None and not Path(cfg.sprites_dir).is_dir():
        return f"--sprites: folder not found: {cfg.sprites_dir}"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        cfg = config_from_args(argv)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    problem = missing_input(cfg)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return App(cfg).run()
    except StoryError as exc:
        logging.getLogger(__name__).error("Story problem: %s", exc)
        return 2
    except (ValueError, OSError) as exc:  # e.g. an unreadable or damaged telemetry log
        logging.getLogger(__name__).error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception:  # last resort: a clear message and a non-zero exit, not a bare traceback
        logging.getLogger(__name__).exception("Unexpected error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
