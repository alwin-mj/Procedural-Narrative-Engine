"""Privacy notice and consent for using the webcam.

The notice is shown before the camera is opened. Saying yes is remembered on this computer
(``~/.narrative-engine/consent.json``, or ``$NARRATIVE_ENGINE_HOME``) until the notice changes;
``--yes`` accepts for a single run without remembering; ``--forget-consent`` deletes the record.
Nothing is asked when no camera is used (``--video``, ``--replay``).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

NOTICE_VERSION = 1

NOTICE = """\
PRIVACY NOTICE

This program uses your webcam to estimate your facial expression while you play.

  * Everything is processed on this computer. No video or images are stored or
    sent anywhere while you play.
  * Optional features that do save something, only when you ask for them:
      --telemetry          labels and numbers (never images)
      narrative-eval collect   your own face pictures, to a local folder
      --record             the window, including your webcam preview, to a file
  * The first run downloads a small face-model file (one time, from Google's
    public MediaPipe model storage). --mp-model avoids that.
  * A facial expression is not the same as a feeling. The estimate can be wrong,
    especially across lighting, skin tones, glasses, ages and cameras.
  * Press X at any time to switch the camera off, and Q to quit.

Do not use this program to judge other people at work, at school, or anywhere
they are not free to say no.
"""


def home_dir() -> Path:
    override = os.environ.get("NARRATIVE_ENGINE_HOME")
    return Path(override) if override else Path.home() / ".narrative-engine"


def consent_file() -> Path:
    return home_dir() / "consent.json"


def has_consent() -> bool:
    """True if this computer has accepted the current notice."""
    try:
        data = json.loads(consent_file().read_text(encoding="utf-8"))
        return int(data.get("version", 0)) >= NOTICE_VERSION
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def record_consent(now: Optional[datetime] = None) -> Path:
    path = consent_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    path.write_text(
        json.dumps({"version": NOTICE_VERSION, "accepted_at": stamp}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def forget_consent() -> bool:
    """Delete the stored consent. Returns True if there was one."""
    path = consent_file()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def ensure_consent(
    accepted_for_this_run: bool = False,
    *,
    input_fn: Callable[[str], str] = input,
    is_tty: Optional[bool] = None,
    out: Callable[[str], None] = print,
) -> bool:
    """Make sure the user has agreed to the notice. Returns False if the camera must stay off."""
    if accepted_for_this_run or has_consent():
        return True
    out(NOTICE)
    interactive = sys.stdin.isatty() if is_tty is None else is_tty
    if not interactive:
        out(
            "The camera needs your consent. Run narrative-engine once in a terminal and answer "
            "the question, or pass --yes to accept for this run."
        )
        return False
    try:
        answer = input_fn("Continue, and remember this choice on this computer? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() in ("y", "yes"):
        record_consent()
        return True
    out("Okay. The camera will not be opened.")
    return False
