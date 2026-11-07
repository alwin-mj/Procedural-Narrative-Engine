"""Session telemetry (JSONL) and replay.

Logs contain only numbers and labels (mood, valence/arousal, scene, timings). **No video or
images are ever recorded.** One JSON object per line::

    {"e":"session_start","schema":1,"participant":"P01","mode":"adaptive",...}
    {"e":"tick","t":1.234,"m":"happy","c":0.8,"b":[x,y,w,h],"s":[h,u,s,n],"v":0.4,"a":0.1}
    {"e":"scene","t":3.6,"to":"market","from":"wake","mood":"neutral","reason":"..."}
    {"e":"session_end","t":42.0,"route":[...],"ending":"radiant","metrics":{...}}

``t`` is seconds since the first event. A log can be replayed through the director
(``--replay``) to compare policies on exactly the same recorded reactions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np

from .config import MOODS
from .perception.detector import Detection

log = logging.getLogger(__name__)

SCHEMA = 1


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", text)[:32] or "anon"


class TelemetryWriter:
    """Append-only JSONL log. Never raises: a full disk must not crash the experience."""

    def __init__(
        self, directory: Path, meta: Dict[str, Any], now: Optional[datetime] = None
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        base = f"session_{stamp}_{_safe_id(str(meta.get('participant', '')))}"
        self.path = directory / f"{base}.jsonl"
        n = 1
        while self.path.exists():  # two runs in the same second must not overwrite each other
            self.path = directory / f"{base}_{n}.jsonl"
            n += 1
        self._f: Optional[TextIO] = open(self.path, "w", encoding="utf-8", buffering=1)
        self.write("session_start", 0.0, schema=SCHEMA, **meta)

    def write(self, kind: str, t: float, **data: Any) -> None:
        if self._f is None:
            return
        rec = {"e": kind, "t": round(float(t), 3), **data}
        try:
            self._f.write(json.dumps(rec, separators=(",", ":"), default=_jsonable) + "\n")
        except (OSError, ValueError) as exc:
            log.warning("telemetry disabled after a write error: %s", exc)
            self.close()

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def tick_record(det: Detection) -> Dict[str, Any]:
    """Compact per-reading record (everything needed to replay the reading)."""
    return {
        "m": det.emotion,
        "c": round(float(det.confidence), 3),
        "b": list(det.box) if det.box is not None else None,
        "s": [round(float(det.scores.get(m, 0.0)), 3) for m in MOODS],
        "v": round(float(det.valence), 3),
        "a": round(float(det.arousal), 3),
    }


# ---------------------------------------------------------------------- reading + replay
@dataclass
class Session:
    meta: Dict[str, Any]
    events: List[Dict[str, Any]]

    @property
    def ticks(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["e"] == "tick"]

    @property
    def scenes(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["e"] == "scene"]

    @property
    def end(self) -> Optional[Dict[str, Any]]:
        ends = [e for e in self.events if e["e"] == "session_end"]
        return ends[-1] if ends else None


def read_session(path: Path) -> Session:
    """Load a telemetry log. Raises ``ValueError`` for files that are not valid sessions."""
    events: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if n == 1:
                        raise ValueError(f"{path}: not a telemetry file ({exc})") from exc
                    log.warning("%s: skipping a damaged line %d (log cut off?)", path, n)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not events or events[0].get("e") != "session_start":
        raise ValueError(f"{path}: missing session_start")
    if events[0].get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported telemetry schema {events[0].get('schema')!r}")
    return Session(events[0], events[1:])


def tick_to_detection(tick: Dict[str, Any]) -> Detection:
    raw = tick.get("b")
    box: Optional[Tuple[int, int, int, int]] = None
    if raw:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    scores = {m: float(v) for m, v in zip(MOODS, tick["s"])}
    return Detection(
        tick["m"], float(tick["c"]), box, scores, float(tick["v"]), float(tick["a"])
    )


class ReplayDetector:
    """Stands in for the camera pipeline: returns the recorded readings in order."""

    supports_calibration = False
    calibrated = False
    active_backend = "replay"

    def __init__(self, ticks: List[Dict[str, Any]]) -> None:
        if not ticks:
            raise ValueError("the telemetry log contains no readings to replay")
        self._ticks = ticks
        self._i = 0

    def detect_stable(self, frame: Any) -> Detection:
        tick = self._ticks[min(self._i, len(self._ticks) - 1)]
        self._i += 1
        return tick_to_detection(tick)

    def close(self) -> None:
        pass


class ReplaySource:
    """Frame source for a replay: blank frames, with time taken from the recording."""

    realtime = False

    def __init__(self, ticks: List[Dict[str, Any]]) -> None:
        self._ticks = ticks
        self._i = 0
        self.finished = False
        self.current_time = 0.0

    @property
    def duration(self) -> float:
        return float(self._ticks[-1]["t"]) if self._ticks else 0.0

    @property
    def fps(self) -> float:
        n = len(self._ticks)
        return (n - 1) / self.duration if n > 1 and self.duration > 0 else 30.0

    def open(self) -> bool:
        self._i, self.finished = 0, False
        return bool(self._ticks)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._i >= len(self._ticks):
            self.finished = True
            return False, None
        self.current_time = float(self._ticks[self._i]["t"])
        self._i += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def reopen(self) -> bool:
        return False

    def release(self) -> None:
        pass
