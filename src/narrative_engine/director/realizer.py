"""Turns a story beat into the text shown on screen.

* :class:`AuthoredRealizer` returns the hand-written line (always works).
* :class:`LLMRealizer` asks a local Ollama model to re-voice that line for the viewer's mood,
  keeping every plot fact. It has a timeout and strict output checks.
* :class:`RealizerWorker` runs the LLM on a background thread with a small cache, so the UI
  never waits: the authored line is shown immediately and swapped for the LLM version when it
  arrives. Failures fall back silently to the authored text and are counted.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Optional, Protocol, Tuple

log = logging.getLogger(__name__)

TONES: Dict[str, str] = {
    "happy": "warm, joyful and playful",
    "uplift": "excited, full of wonder and anticipation",
    "sad": "gentle, quiet and comforting",
    "neutral": "calm and observant",
}
MIN_CHARS = 20
MAX_CHARS = 300


@dataclass(frozen=True)
class BeatRequest:
    """Everything needed to realize one beat."""

    authored: str  # the hand-written line: the source of truth for plot facts
    mood: str
    valence: float = 0.0
    arousal: float = 0.0
    title: str = ""
    protagonist: str = ""
    location: str = ""
    recent: Tuple[str, ...] = ()  # last few lines shown, oldest first


class Realizer(Protocol):
    def realize(self, req: BeatRequest) -> Optional[str]:
        """Return text for the beat, or ``None`` to use the authored line."""


class AuthoredRealizer:
    def realize(self, req: BeatRequest) -> Optional[str]:
        return req.authored


# ---------------------------------------------------------------------- prompt + checks
def build_prompt(req: BeatRequest) -> str:
    who = f" The protagonist is {req.protagonist}." if req.protagonist else ""
    where = f" Setting: {req.location}." if req.location else ""
    recent = " ".join(req.recent[-3:]) or "(this is the first scene)"
    return (
        f'You are the narrator of a short interactive story called "{req.title or "Untitled"}".'
        f"{who}{where}\n"
        f"Story so far: {recent}\n\n"
        f"Scene to narrate: {req.authored}\n\n"
        f"Re-tell this scene in 1-2 sentences (under 45 words) in a {TONES.get(req.mood, 'calm')} "
        f"tone. Keep every plot fact and every named character or object. Do not add new "
        f"events. Write only the narration, with no quotes, headings or lists."
    )


def clean_output(text: str, protagonist: str = "", max_chars: int = MAX_CHARS) -> Optional[str]:
    """Validate and tidy model output. Returns ``None`` if it should be rejected."""
    if not text or re.search(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s", text):
        return None  # empty, or a bulleted / numbered list instead of narration
    out = re.sub(r"\s+", " ", text).strip().strip("\"'`“”‘’ ")
    if re.match(r"^(sure|here|certainly|okay)[,:! ]", out, re.IGNORECASE):
        return None  # chatty preamble instead of narration
    if len(out) > max_chars:
        cut = out[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if end < MIN_CHARS:
            return None
        out = cut[: end + 1]
    if len(out) < MIN_CHARS:
        return None
    if protagonist and protagonist.lower() not in out.lower():
        return None  # lost the hero: likely drifted from the scene
    if any(ch in out for ch in ("#", "*", "\n", "•")):
        return None
    return out


class LLMRealizer:
    """Local Ollama client (``/api/generate``, non-streaming). Stdlib only."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_s: float = 6.0,
        warmup_timeout_s: float = 30.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.timeout_s = timeout_s
        self.warmup_timeout_s = warmup_timeout_s  # the first call loads the model: allow longer
        self._opener = opener
        self._warm = False

    def realize(self, req: BeatRequest) -> Optional[str]:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": build_prompt(req),
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 100},
            }
        ).encode("utf-8")
        http_req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        timeout = self.timeout_s if self._warm else max(self.timeout_s, self.warmup_timeout_s)
        with self._opener(http_req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self._warm = True
        return clean_output(str(payload.get("response", "")), req.protagonist)


# ---------------------------------------------------------------------- async worker
class RealizerWorker:
    """Background LLM calls + cache. ``get`` and ``submit`` never block the UI thread."""

    FAILURES_BEFORE_PAUSE = 3
    PAUSE_S = 30.0

    def __init__(
        self,
        realizer: Any,
        clock: Callable[[], float] = time.monotonic,
        cache_size: int = 64,
    ) -> None:
        self._realizer = realizer
        self._clock = clock
        self._cache: "OrderedDict[Hashable, str]" = OrderedDict()
        self._cache_size = cache_size
        self._pending: set = set()
        self._queue: "queue.Queue[Optional[Tuple[Hashable, BeatRequest]]]" = queue.Queue(maxsize=8)
        self._lock = threading.Lock()
        self._failures = 0
        self._paused_until = 0.0
        self.requests = 0  # calls that reached the model
        self.successes = 0
        self.fallbacks = 0  # calls that failed, timed out or were rejected
        self._thread = threading.Thread(target=self._run, name="realizer", daemon=True)
        self._thread.start()

    # ---- UI-thread API
    def get(self, key: Hashable) -> Optional[str]:
        with self._lock:
            text = self._cache.get(key)
            if text is not None:
                self._cache.move_to_end(key)
            return text

    def submit(self, key: Hashable, req: BeatRequest) -> bool:
        """Queue a request unless cached, already pending, paused or the queue is full."""
        with self._lock:
            if key in self._cache or key in self._pending:
                return False
            if self._clock() < self._paused_until:
                return False
            try:
                self._queue.put_nowait((key, req))
            except queue.Full:
                return False
            self._pending.add(key)
            return True

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.requests if self.requests else 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "fallbacks": self.fallbacks,
            "fallback_rate": round(self.fallback_rate, 3),
        }

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Block until queued work is done (used by tests)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ---- worker thread
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            key, req = item
            text: Optional[str] = None
            ok = False
            self.requests += 1
            try:
                text = self._realizer.realize(req)
                ok = text is not None
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                log.debug("LLM request failed: %s", exc)
            except Exception as exc:  # never let the worker thread die
                log.warning("LLM realizer error: %s", exc)
            with self._lock:
                self._pending.discard(key)
                if ok and text:
                    self._cache[key] = text
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                    self._failures = 0
                    self.successes += 1
                else:
                    self.fallbacks += 1
                    self._failures += 1
                    if self._failures >= self.FAILURES_BEFORE_PAUSE:
                        self._paused_until = self._clock() + self.PAUSE_S
                        self._failures = 0
                        log.warning(
                            "LLM unavailable after repeated failures; pausing for %.0fs "
                            "(authored text is used meanwhile)",
                            self.PAUSE_S,
                        )
