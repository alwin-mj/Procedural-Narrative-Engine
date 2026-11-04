"""Main loop: camera -> perception -> director -> renderer."""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set

import cv2
import numpy as np

from .config import MOODS, Config
from .director import (
    Affect,
    Decision,
    DramaManager,
    StoryGraph,
    StorySession,
    choose_advance_emotion,
    load_story,
)
from .director.realizer import BeatRequest, LLMRealizer, RealizerWorker
from .perception import EmotionDetector, FaceMotion, MotionDetector
from .render import GameWindow, PlayerAnimation

log = logging.getLogger(__name__)

PROMPT_TEXT = "Press S to start the adventure!"
NOTICE_SECONDS = 2.0
KEY_QUIT = ord("q")
KEY_START = ord("s")
KEY_NEXT = ord(" ")
KEY_CALIBRATE = ord("c")


class App:
    """Owns all components and the per-frame update logic.

    ``detector``, ``story``, ``realizer``, ``clock`` and ``rng`` can be injected, which keeps
    the logic testable without a camera, a language model, or real time.
    """

    def __init__(
        self,
        cfg: Config,
        detector: Optional[Any] = None,
        story: Optional[StoryGraph] = None,
        clock: Callable[[], float] = time.monotonic,
        realizer: Optional[Any] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.cfg = cfg
        self._clock = clock
        self._rng = rng or random.Random()
        self.graph: StoryGraph = story or load_story(cfg.story_path)
        self.session = StorySession(self.graph)
        self.drama = DramaManager(self.session)
        self.detector = detector or EmotionDetector(
            cfg.backend,
            cfg.smoothing_buffer,
            cfg.smoothing_window,
            hysteresis_margin=cfg.hysteresis_margin,
            hysteresis_frames=cfg.hysteresis_frames,
            affect_alpha=cfg.affect_smoothing,
            mediapipe_model=cfg.mediapipe_model,
        )
        threshold = cfg.resolved_motion_threshold
        self.motion = (
            FaceMotion(threshold) if cfg.motion_mode == "face" else MotionDetector(threshold)
        )

        anim = (
            PlayerAnimation(cfg.sprites_dir, size=cfg.sprite_size, fps=cfg.sprite_fps)
            if cfg.sprites_dir
            else None
        )
        self.window = GameWindow(cfg.width, cfg.height, player_anim=anim)

        self.worker: Optional[RealizerWorker] = None
        if realizer is not None:
            self.worker = RealizerWorker(realizer, clock=clock)
        elif cfg.llm:
            llm = LLMRealizer(cfg.llm, cfg.llm_host, cfg.llm_timeout_s)
            self.worker = RealizerWorker(llm, clock=clock)
        self._requested: Set[Any] = set()
        self._recent: Deque[str] = deque(maxlen=3)

        self.history: Deque[str] = deque(maxlen=cfg.history_len)
        self.active = False
        self.last_decision: Optional[Decision] = None
        self._last_advance = 0.0
        self._last_chosen: Optional[str] = None
        self._last_scores: Dict[str, float] = {}
        self._latest_scores: Dict[str, float] = {}
        self._latest_emo = "neutral"
        self._latest_affect = Affect()
        self._random_mood = "neutral"
        self._acc_v = self._acc_a = 0.0  # affect accumulated during the current scene
        self._acc_n = 0

        # mood-shift tracking: a mood must persist for mood_dwell_s to count as a shift
        self._candidate = "neutral"
        self._candidate_since = 0.0
        self._shown_emo = "neutral"
        self._reaction_until = 0.0

        # calibration / transient messages
        self._calib_until: Optional[float] = None
        self._notice = ""
        self._notice_until = 0.0

    # ------------------------------------------------------------------ story control
    def _scene_affect(self) -> Affect:
        """Mean valence/arousal over the scene being played (latest reading if none yet)."""
        if self._acc_n:
            return Affect(self._acc_v / self._acc_n, self._acc_a / self._acc_n)
        return self._latest_affect

    def _display_mood(self, emo: str) -> str:
        """Which mood variant to show: live (adaptive), always neutral (fixed), or random."""
        mode = self.cfg.director_mode
        if mode == "fixed":
            return "neutral"
        if mode == "random":
            return self._random_mood
        return emo

    def _advance(self, now: float) -> None:
        s = self.session
        if s.finished:
            return
        mode = self.cfg.director_mode
        affect = self._scene_affect()
        if mode == "fixed":
            mood = "neutral"
        elif mode == "random":
            mood = self._random_mood = self._rng.choice(MOODS)
        else:
            mood = choose_advance_emotion(
                self._latest_scores, self._last_chosen, self._last_scores, self.cfg.mood_bias
            )

        if mode == "fixed":
            decision = self.drama.decide_fixed(mood, affect)
        elif mode == "random":
            decision = self.drama.decide_random(mood, affect, self._rng)
        else:
            decision = self.drama.decide(mood, affect)

        if decision.to_id is None:
            s.finish(mood, affect)
        else:
            s.advance(decision.to_id, mood, affect)
            self._recent.append(s.get_line(mood))
        self.last_decision = decision
        self._last_chosen = mood
        self._last_scores = dict(self._latest_scores)
        self._last_advance = now
        self._acc_v = self._acc_a = 0.0
        self._acc_n = 0
        log.info("director: %s", decision.reason)
        if s.finished:
            log.info("ending: %s", s.ending_id)
        self._prefetch(self._display_mood(self._latest_emo), affect)

    def _start(self, now: float) -> None:
        self.session.reset()
        self.active = True
        self.last_decision = None
        self._last_advance = now
        self._last_chosen = None
        self._last_scores = {}
        self._acc_v = self._acc_a = 0.0
        self._acc_n = 0
        self._reaction_until = 0.0
        self._shown_emo = self._latest_emo
        self._recent.clear()
        self._requested.clear()
        if self.cfg.director_mode == "random":
            self._random_mood = self._rng.choice(MOODS)
        if (
            self.cfg.calibrate_on_start
            and getattr(self.detector, "supports_calibration", False)
            and not getattr(self.detector, "calibrated", False)
        ):
            self.start_calibration(now)
        self._prefetch(self._display_mood(self._latest_emo), self._latest_affect)

    # ------------------------------------------------------------------ LLM text
    def _request(self, text: str, mood: str, location: str, affect: Affect) -> BeatRequest:
        return BeatRequest(
            authored=text,
            mood=mood,
            valence=affect.valence,
            arousal=affect.arousal,
            title=self.graph.title,
            protagonist=self.graph.protagonist,
            location=location,
            recent=tuple(self._recent),
        )

    def _prefetch(self, mood: str, affect: Affect) -> None:
        """Ask the LLM for the likely next scenes now, so their text is ready when needed."""
        s = self.session
        if self.worker is None or s.finished or s.is_terminal:
            return
        for edge in s.candidates(mood, affect)[:3]:
            text = s.preview_line(edge.to, mood, affect)
            key = (edge.to, mood, text)
            if key not in self._requested:
                loc = self.graph.nodes[edge.to].location
                if self.worker.submit(key, self._request(text, mood, loc, affect)):
                    self._requested.add(key)

    def _scene_text(self, emo: str) -> str:
        """Authored text now; the LLM version replaces it as soon as it is available."""
        mood = self._display_mood(emo)
        s = self.session
        base = s.get_line(mood)
        if self.worker is None or s.finished:
            return base
        key = (s.current_id, mood, base)
        cached = self.worker.get(key)
        if cached:
            return cached
        if key not in self._requested:
            req = self._request(base, mood, s.node.location, self._scene_affect())
            if self.worker.submit(key, req):
                self._requested.add(key)
        return base

    # ------------------------------------------------------------------ calibration
    def start_calibration(self, now: float) -> bool:
        """Begin the neutral-face calibration. Returns False if the backend cannot calibrate."""
        if not getattr(self.detector, "supports_calibration", False):
            self._set_notice("Calibration needs the MediaPipe backend", now)
            return False
        self.detector.begin_calibration()
        self._calib_until = now + self.cfg.calibration_s
        return True

    def _finish_calibration(self, now: float) -> None:
        ok = self.detector.finish_calibration()
        self._calib_until = None
        self._last_advance = now
        self._set_notice("Calibrated!" if ok else "No face seen. Press C to retry", now)

    def _set_notice(self, text: str, now: float) -> None:
        self._notice, self._notice_until = text, now + NOTICE_SECONDS

    # ------------------------------------------------------------------ mood tracking
    def _track_mood(self, emo: str, conf: float, now: float) -> None:
        if emo != self._candidate:
            self._candidate, self._candidate_since = emo, now
        stable_for = now - self._candidate_since
        if not self.active:
            self._shown_emo = emo
        elif (
            emo != self._shown_emo
            and stable_for >= self.cfg.mood_dwell_s
            and conf > self.cfg.reaction_min_confidence
        ):
            self._shown_emo = emo
            self._reaction_until = now + self.cfg.reaction_display_s

    def _director_note(self) -> str:
        if not self.active:
            return ""
        s = self.session
        if s.finished:
            return f"ending: {s.ending_id}"
        head = f"scene: {s.current_id} (tension {s.node.tension:.2f})"
        if self.last_decision is not None:
            return f"{head} | {self.last_decision.reason}"
        return head

    # ------------------------------------------------------------------ one frame
    def step(self, frame: np.ndarray) -> np.ndarray:
        """Process one camera frame and return the composed output image."""
        cfg = self.cfg
        now = self._clock()
        det = self.detector.detect_stable(frame)
        emo, conf, box, scores = det.emotion, det.confidence, det.box, det.scores
        walking = (
            self.motion.update(box) if cfg.motion_mode == "face" else self.motion.update(frame)
        )
        self._latest_scores, self._latest_emo = scores, emo
        self._latest_affect = Affect(det.valence, det.arousal)
        self.history.append(emo)

        calibrating = self._calib_until is not None
        if calibrating and now >= self._calib_until:  # type: ignore[operator]
            self._finish_calibration(now)
            calibrating = False

        if not calibrating:
            if self.active and box is not None:
                self._acc_v += det.valence
                self._acc_a += det.arousal
                self._acc_n += 1
            if self.active and box is not None and conf > cfg.min_confidence:
                if now - self._last_advance > cfg.advance_interval_s:
                    self._advance(now)
            self._track_mood(emo, conf, now)

        if calibrating:
            remaining = math.ceil(self._calib_until - now)  # type: ignore[operator]
            text = f"Hold a neutral face... {remaining}"
        elif now < self._notice_until:
            text = self._notice
        elif not self.active:
            text = PROMPT_TEXT
        else:
            text = self._scene_text(emo)
            if now < self._reaction_until and not self.session.finished:
                text = f"A shift: you seem {emo}. {text}"

        return self.window.render(
            frame,
            emo,
            conf,
            box,
            self.active,
            text,
            list(self.history),
            walking=walking,
            backend=getattr(self.detector, "active_backend", ""),
            valence=det.valence,
            arousal=det.arousal,
            director_note=self._director_note(),
        )

    def handle_key(self, key: int) -> bool:
        """Handle a key press. Returns False when the app should quit."""
        if key == KEY_QUIT:
            return False
        now = self._clock()
        if key == KEY_START:
            self._start(now)
        elif key == KEY_NEXT and self.active:
            self._advance(now)
        elif key == KEY_CALIBRATE:
            self.start_calibration(now)
        return True

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            log.error(
                "Cannot open camera %d. Close other apps using it or try --camera 1.",
                self.cfg.camera_index,
            )
            return 1
        log.info("Running. Keys: S start, C calibrate, SPACE next scene, Q quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                cv2.imshow(self.cfg.window_title, self.step(frame))
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            close = getattr(self.detector, "close", None)
            if callable(close):
                close()
            if self.worker is not None:
                log.info("LLM stats: %s", self.worker.stats())
                self.worker.close()
        return 0
