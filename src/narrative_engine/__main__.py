"""Entry point: ``python -m narrative_engine`` or the ``narrative-engine`` command."""

from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence

from .app import App
from .config import config_from_args
from .director import StoryError


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        cfg = config_from_args(argv)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
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
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
