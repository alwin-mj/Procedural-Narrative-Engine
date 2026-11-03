"""Main loop: camera -> perception -> director -> renderer."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import cv2
import numpy as np

from .config import Config
from .director import Story, choose_advance_emotion, load_story
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

    ``detector``, ``story`` and ``clock`` can be injected, which keeps the logic testable
    without a camera or real time.
    """

    def __init__(
        self,
        cfg: Config,
        detector: Optional[Any] = None,
        story: Optional[Story] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._clock = clock
        self.story: Story = story or load_story(cfg.story_path)
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

        self.history: Deque[str] = deque(maxlen=cfg.history_len)
        self.active = False
        self._last_advance = 0.0
        self._last_chosen: Optional[str] = None
        self._last_scores: Dict[str, float] = {}
        self._latest_scores: Dict[str, float] = {}
        self._latest_emo = "neutral"

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
    def _advance(self, scores: Dict[str, float], now: float) -> str:
        chosen = choose_advance_emotion(
            scores, self._last_chosen, self._last_scores, happy_bias=self.cfg.happy_bias
        )
        text = self.story.advance(chosen)
        self._last_chosen = chosen
        self._last_scores = dict(scores)
        self._last_advance = now
        log.debug("advanced to scene %d as %r", self.story.scene_idx, chosen)
        return text

    def _start(self, now: float) -> None:
        self.story.reset()
        self.active = True
        self._last_advance = now
        self._last_chosen = None
        self._last_scores = {}
        self._reaction_until = 0.0
        self._shown_emo = self._latest_emo
        if (
            self.cfg.calibrate_on_start
            and getattr(self.detector, "supports_calibration", False)
            and not getattr(self.detector, "calibrated", False)
        ):
            self.start_calibration(now)

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
        self.history.append(emo)

        calibrating = self._calib_until is not None
        if calibrating and now >= self._calib_until:  # type: ignore[operator]
            self._finish_calibration(now)
            calibrating = False

        if not calibrating:
            if self.active and box is not None and conf > cfg.min_confidence:
                if now - self._last_advance > cfg.advance_interval_s:
                    self._advance(scores, now)
            self._track_mood(emo, conf, now)

        if calibrating:
            remaining = math.ceil(self._calib_until - now)  # type: ignore[operator]
            text = f"Hold a neutral face... {remaining}"
        elif now < self._notice_until:
            text = self._notice
        elif not self.active:
            text = PROMPT_TEXT
        else:
            text = self.story.get_line(emo)
            if now < self._reaction_until:
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
        )

    def handle_key(self, key: int) -> bool:
        """Handle a key press. Returns False when the app should quit."""
        if key == KEY_QUIT:
            return False
        now = self._clock()
        if key == KEY_START:
            self._start(now)
        elif key == KEY_NEXT and self.active:
            self._advance(self._latest_scores, now)
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
        return 0
