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
    names_hero = bool(req.protagonist) and req.protagonist.lower() in req.authored.lower()
    name_rule = f" Use the name {req.protagonist} at least once." if names_hero else ""
    return (
        f'You are the narrator of a short interactive story called "{req.title or "Untitled"}".'
        f"{who}{where}\n"
        f"Story so far: {recent}\n\n"
        f"Scene to narrate: {req.authored}\n\n"
        f"Re-tell this scene in 1-2 sentences (under 45 words) in a {TONES.get(req.mood, 'calm')} "
        f"tone. Keep every plot fact and every named character or object. Do not add new "
        f"events.{name_rule} Write only the narration, with no quotes, headings or lists."
    )


class OutputRejected(ValueError):
    """The model answered, but the answer failed the checks (as opposed to a failed request)."""

    def __init__(self, reason: str, sample: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.sample = sample


def check_output(
    text: str,
    protagonist: str = "",
    authored: Optional[str] = None,
    max_chars: int = MAX_CHARS,
) -> Tuple[Optional[str], Optional[str]]:
    """Validate and tidy model output. Returns ``(clean_text, None)`` or ``(None, reason)``.

    The hero's name is only required when the authored line it retells contains that name.
    Many authored lines say "he", and a faithful retelling of those must not be rejected
    for leaving the name out. With ``authored=None`` the name is always required.
    """
    if not text or not text.strip():
        return None, "empty answer"
    if re.search(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s", text):
        return None, "bulleted or numbered list instead of narration"
    out = re.sub(r"\s+", " ", text).strip().strip("\"'`“”‘’ ")
    if re.match(r"^(sure|here|certainly|okay)[,:! ]", out, re.IGNORECASE):
        return None, "chatty preamble instead of narration"
    if re.match(r"^(i'?m sorry|i am sorry|sorry|i cannot|i can't|as an ai)\b", out, re.IGNORECASE):
        return None, "refusal instead of narration"
    if len(out) > max_chars:
        cut = out[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if end < MIN_CHARS:
            return None, f"longer than {max_chars} characters with no sentence break"
        out = cut[: end + 1]
    if len(out) < MIN_CHARS:
        return None, f"shorter than {MIN_CHARS} characters"
    needs_name = bool(protagonist) and (authored is None or protagonist.lower() in authored.lower())
    if needs_name and protagonist.lower() not in out.lower():
        return None, f"lost the hero's name ({protagonist!r}) that the scene mentions"
    if any(ch in out for ch in ("#", "*", "•")):
        return None, "markdown formatting"
    return out, None


def clean_output(
    text: str,
    protagonist: str = "",
    max_chars: int = MAX_CHARS,
    authored: Optional[str] = None,
) -> Optional[str]:
    """Like :func:`check_output` but returns only the cleaned text (``None`` if rejected)."""
    return check_output(text, protagonist, authored, max_chars)[0]


def check_ollama(
    host: str,
    model: str,
    timeout_s: float = 2.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Tuple[bool, str]:
    """Is an Ollama server reachable and does it have ``model``? Returns ``(ok, message)``."""
    url = host.rstrip("/") + "/api/tags"
    try:
        with opener(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"Ollama is not reachable at {host} ({reason}). Start it with: ollama serve"
    names = [str(m.get("name", "")) for m in payload.get("models", []) if isinstance(m, dict)]
    if any(n == model or n.startswith(model + ":") for n in names):
        return True, f"Ollama is running and has model {model!r}"
    have = ", ".join(names) or "none"
    return False, (
        f"Ollama is running but model {model!r} is not installed (have: {have}). "
        f"Run: ollama pull {model}"
    )


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
        raw = str(payload.get("response", ""))
        cleaned, reason = check_output(raw, req.protagonist, req.authored)
        if cleaned is None:
            raise OutputRejected(reason or "rejected", raw)
        return cleaned


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
        self.errors = 0  # the request itself failed (server down, timeout, HTTP error)
        self.rejected = 0  # the model answered but the answer failed the checks
        self.fallbacks = 0  # errors + rejections: the authored text was used instead
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
            "errors": self.errors,
            "rejected": self.rejected,
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
            outcome = "error"
            self.requests += 1
            try:
                text = self._realizer.realize(req)
                outcome = "ok" if text else "rejected"
            except OutputRejected as exc:
                outcome = "rejected"
                if self.rejected < 3:  # say why, so a bad prompt or model is easy to spot
                    log.warning("LLM answer rejected (%s): %.120r", exc.reason, exc.sample)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                if self.errors < 3:
                    log.warning("LLM request failed: %s", getattr(exc, "reason", exc))
            except Exception as exc:  # never let the worker thread die
                log.warning("LLM realizer error: %s", exc)
            with self._lock:
                self._pending.discard(key)
                if outcome == "ok" and text:
                    self._cache[key] = text
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                    self._failures = 0
                    self.successes += 1
                elif outcome == "rejected":
                    self.rejected += 1  # the server is fine, so this never pauses the worker
                    self.fallbacks += 1
                    self._failures = 0
                else:
                    self.errors += 1
                    self.fallbacks += 1
                    self._failures += 1
                    if self._failures >= self.FAILURES_BEFORE_PAUSE:
                        self._paused_until = self._clock() + self.PAUSE_S
                        self._failures = 0
                        log.warning(
                            "LLM unreachable after repeated failures; pausing for %.0fs "
                            "(authored text is used meanwhile)",
                            self.PAUSE_S,
                        )
