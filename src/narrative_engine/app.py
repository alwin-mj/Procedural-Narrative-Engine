"""Main loop: camera -> perception -> director -> renderer."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2

from .config import Config
from .director import Story, choose_advance_emotion, load_story
from .perception import EmotionDetector, MotionDetector
from .render import GameWindow, PlayerAnimation

log = logging.getLogger(__name__)

PROMPT_TEXT = "Press S to start the adventure!"
KEY_QUIT = ord("q")
KEY_START = ord("s")
KEY_NEXT = ord(" ")


class App:
    """Owns all components and the per-frame update logic."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.story: Story = load_story(cfg.story_path)
        self.detector = EmotionDetector(cfg.backend, cfg.smoothing_buffer, cfg.smoothing_window)
        self.motion = MotionDetector(cfg.motion_threshold)

        anim = (
            PlayerAnimation(cfg.sprites_dir, size=cfg.sprite_size, fps=cfg.sprite_fps)
            if cfg.sprites_dir
            else None
        )
        self.window = GameWindow(cfg.width, cfg.height, player_anim=anim)

        self.history: Deque[str] = deque(maxlen=cfg.history_len)
        self.active = False
        self._last_advance = 0.0
        self._last_emo = "neutral"
        self._reaction_until = 0.0
        self._last_chosen: Optional[str] = None
        self._last_scores: Dict[str, float] = {}
        self._latest_scores: Dict[str, float] = {}

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

    # ------------------------------------------------------------------ one frame
    def step(self, frame):
        """Process one camera frame and return the composed output image."""
        cfg = self.cfg
        walking = self.motion.update(frame)
        det = self.detector.detect_stable(frame)
        emo, conf, box, scores = det
        self._latest_scores = scores
        self.history.append(emo)

        now = time.monotonic()
        face_present = box is not None

        # Auto-advance while a face is visible and the reading is confident enough.
        if (
            self.active
            and face_present
            and now - self._last_advance > cfg.advance_interval_s
            and conf > cfg.min_confidence
        ):
            self._advance(scores, now)

        # Briefly flag a mood change without skipping the scene.
        if self.active and emo != self._last_emo and conf > cfg.reaction_min_confidence:
            self._reaction_until = now + cfg.reaction_display_s
            self._last_advance = now
        self._last_emo = emo

        if not self.active:
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
            backend=self.detector.active_backend,
        )

    def handle_key(self, key: int) -> bool:
        """Handle a key press. Returns False when the app should quit."""
        if key == KEY_QUIT:
            return False
        now = time.monotonic()
        if key == KEY_START:
            self._start(now)
        elif key == KEY_NEXT and self.active:
            self._advance(self._latest_scores, now)
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
        log.info("Running. Keys: S start, SPACE next scene, Q quit.")
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
        return 0
