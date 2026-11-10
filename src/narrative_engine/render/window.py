"""Frame composition: emotion-coloured background, particles, character, HUD, story text."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import MOODS
from .player_animation import PlayerAnimation

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]
Color = Tuple[int, int, int]

# (top, bottom) gradient colours in BGR, per mood.
GRADIENTS: Dict[str, Tuple[Color, Color]] = {
    "happy": ((60, 200, 100), (20, 100, 40)),
    "uplift": ((60, 200, 255), (10, 60, 120)),
    "sad": ((120, 100, 160), (40, 30, 60)),
    "neutral": ((140, 140, 150), (60, 60, 70)),
}


def wrap_text(text: str, width: int) -> List[str]:
    """Greedy word wrap to at most ``width`` characters per line."""
    if not text:
        return [""]
    lines: List[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines


class GameWindow:
    """Draws one frame of the experience; holds only cosmetic state (particles, orb position)."""

    def __init__(
        self, width: int = 960, height: int = 540, player_anim: Optional[PlayerAnimation] = None
    ) -> None:
        self.w = width
        self.h = height
        self.player_anim = player_anim
        self.anim_pos = 60.0
        self.anim_dir = 1

        self._rng = np.random.default_rng()
        self._vignette = self._make_vignette()
        self._backgrounds: Dict[str, np.ndarray] = {}  # cached gradient * vignette per mood

        now = time.time()
        self.particles = [
            {
                "x": float(self._rng.integers(0, self.w)),
                "y": float(self._rng.integers(0, max(1, self.h // 2))),
                "vx": float(self._rng.uniform(-0.3, 0.3)),
                "vy": float(self._rng.uniform(-0.1, 0.1)),
                "r": int(self._rng.integers(1, 4)),
                "life": float(self._rng.uniform(3.0, 8.0)),
                "t": now - float(self._rng.uniform(0, 8.0)),
            }
            for _ in range(40)
        ]

    # ------------------------------------------------------------------ background
    def _make_vignette(self) -> np.ndarray:
        yv, xv = np.ogrid[0 : self.h, 0 : self.w]
        cx, cy = self.w / 2, self.h / 2
        dist = ((xv - cx) / cx) ** 2 + ((yv - cy) / cy) ** 2
        return (1.0 - np.clip(dist * 0.6, 0.0, 0.8)).astype(np.float32)

    def _background(self, emo: str) -> np.ndarray:
        """Vertical gradient with vignette, computed once per mood and reused."""
        key = emo if emo in GRADIENTS else "neutral"
        if key not in self._backgrounds:
            top = np.array(GRADIENTS[key][0], dtype=np.float32)
            bottom = np.array(GRADIENTS[key][1], dtype=np.float32)
            t = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None]
            column = top * (1 - t) + bottom * t  # (h, 3)
            grad = np.broadcast_to(column[:, None, :], (self.h, self.w, 3))
            self._backgrounds[key] = (grad * self._vignette[..., None]).astype(np.uint8)
        return self._backgrounds[key].copy()

    # ------------------------------------------------------------------ pieces
    def _draw_particles(self, canvas: np.ndarray, emo: str) -> None:
        now = time.time()
        bright = emo in ("happy", "uplift")
        color: Color = (255, 255, 220) if bright else (200, 200, 220)
        for p in self.particles:
            life_ratio = max(0.0, 1.0 - (now - p["t"]) / p["life"])
            p["x"] += p["vx"]
            p["y"] += p["vy"] - (0.02 if emo == "uplift" else 0.0)
            if p["x"] < -10:
                p["x"] = self.w + 10
            if p["x"] > self.w + 10:
                p["x"] = -10
            if p["y"] < 0:
                p["y"] = self.h // 2
            if p["y"] > self.h:
                p["y"] = 0
            radius = max(1, int(p["r"] + life_ratio * 2))
            cv2.circle(canvas, (int(p["x"]), int(p["y"])), radius, color, -1)

    def _draw_player(self, canvas: np.ndarray, walking: bool) -> np.ndarray:
        anim = self.player_anim
        if anim is not None and anim.has_frames:
            frm = anim.get_frame()
            if frm is not None:
                px = (self.w // 2) - (frm.shape[1] // 2)
                py = self.h // 2
                try:
                    canvas = anim.overlay_png(canvas, frm, px, py)
                except (ValueError, cv2.error) as exc:
                    log.debug("sprite overlay failed: %s", exc)
            return canvas

        # Default character: a glowing orb that moves faster when the viewer moves.
        self.anim_pos += self.anim_dir * (6 if walking else 3)
        if self.anim_pos > self.w - 80 or self.anim_pos < 80:
            self.anim_dir *= -1
        center = (int(self.anim_pos), self.h // 3)
        for radius, alpha in ((40, 0.06), (28, 0.12)):
            overlay = canvas.copy()
            cv2.circle(overlay, center, radius, (255, 255, 255), -1)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
        return canvas

    def _draw_webcam_preview(
        self, canvas: np.ndarray, frame: Optional[np.ndarray], box: Optional[Box], x0: int, y0: int
    ) -> int:
        """Circular webcam thumbnail. Returns its height (for placing text below it)."""
        default_h = 90
        if frame is None:
            return default_h
        fh, fw = frame.shape[:2]
        scale = min((self.w // 5) / fw, (self.h // 5) / fh)
        sw, sh = max(90, int(fw * scale)), max(70, int(fh * scale))
        try:
            small = cv2.resize(frame, (sw, sh))
            mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.circle(mask, (sw // 2, sh // 2), min(sw, sh) // 2, 255, -1)
            region = canvas[y0 : y0 + sh, x0 : x0 + sw]
            region[:] = np.where(mask[..., None] == 255, small, region)
            cv2.circle(
                canvas, (x0 + sw // 2, y0 + sh // 2), min(sw, sh) // 2 + 3, (255, 255, 255), 2
            )
            if box is not None:
                bx, by, bw, bh = box
                sx = int((bx + bw * 0.5) * sw / fw)
                sy = int((by + bh * 0.5) * sh / fh)
                cv2.circle(canvas, (x0 + sx, y0 + sy), 6, (0, 255, 255), -1)
            return sh
        except (ValueError, cv2.error) as exc:
            log.debug("webcam preview failed: %s", exc)
            return default_h

    def _draw_history(self, canvas: np.ndarray, history: Sequence[str]) -> None:
        counts = Counter(history)
        for i, mood in enumerate(MOODS):
            x, y = self.w - 180, 20 + i * 30
            cv2.rectangle(canvas, (x, y), (x + 120, y + 22), (30, 30, 30), -1)
            bar = min(110, counts.get(mood, 0) * 10)
            cv2.rectangle(canvas, (x + 6, y + 4), (x + 6 + bar, y + 18), (200, 200, 200), -1)
            cv2.putText(
                canvas, mood, (x - 70, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1
            )

    def _draw_affect_pad(
        self, canvas: np.ndarray, x: int, y: int, size: int, valence: float, arousal: float
    ) -> None:
        """Small valence (x) / arousal (y) plot with a dot at the current estimate."""
        cv2.rectangle(canvas, (x, y), (x + size, y + size), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + size, y + size), (200, 200, 200), 1)
        cv2.line(canvas, (x + size // 2, y), (x + size // 2, y + size), (90, 90, 90), 1)
        cv2.line(canvas, (x, y + size // 2), (x + size, y + size // 2), (90, 90, 90), 1)
        v = max(-1.0, min(1.0, valence))
        a = max(-1.0, min(1.0, arousal))
        dx = int(x + (v + 1.0) / 2.0 * size)
        dy = int(y + (1.0 - (a + 1.0) / 2.0) * size)
        cv2.circle(canvas, (dx, dy), 5, (0, 255, 255), -1)
        label = f"V {valence:+.2f}  A {arousal:+.2f}"
        cv2.putText(
            canvas, label, (x, y + size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

    def _draw_director_note(self, canvas: np.ndarray, note: str) -> None:
        """Small right-aligned text block explaining the director's latest decision."""
        x, y = max(12, self.w - 330), 170
        for i, line in enumerate(wrap_text(note, 44)[:5]):
            cv2.putText(
                canvas, line, (x, y + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215, 215, 215), 1
            )

    def _draw_camera_indicator(self, canvas: np.ndarray, on: bool) -> None:
        """Red dot + "CAMERA ON" (or grey "CAMERA OFF"). Always drawn, even in study mode."""
        label = "CAMERA ON" if on else "CAMERA OFF"
        colour = (0, 0, 230) if on else (150, 150, 150)  # BGR
        x, y = self.w // 2 - 60, 24
        cv2.circle(canvas, (x, y - 5), 6, colour, -1)
        cv2.putText(
            canvas,
            label,
            (x + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_story_box(self, canvas: np.ndarray, story_text: str) -> Tuple[np.ndarray, int]:
        """Draw the text box; returns the canvas and the box's top edge (y)."""
        lines = wrap_text(story_text, 48)
        box_h = 26 * len(lines) + 20
        box_w = self.w - 40
        x0, y0 = 20, self.h - box_h - 20
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (10, 10, 10), -1)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (x0 + 18, y0 + 22 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
        return canvas, y0

    # ------------------------------------------------------------------ main entry
    def render(
        self,
        frame: Optional[np.ndarray],
        emo: str,
        conf: float,
        box: Optional[Box],
        active: bool,
        story_text: str,
        history: Optional[Sequence[str]] = None,
        walking: bool = False,
        backend: str = "",
        valence: float = 0.0,
        arousal: float = 0.0,
        director_note: str = "",
        stats: str = "",
        debug: bool = True,
        camera_on: bool = True,
    ) -> np.ndarray:
        canvas = self._background(emo)
        self._draw_particles(canvas, emo)
        canvas = self._draw_player(canvas, walking)

        web_x, web_y = 12, 12
        web_h = self._draw_webcam_preview(canvas, frame, box, web_x, web_y)

        if history:
            self._draw_history(canvas, history)
        self._draw_camera_indicator(canvas, camera_on)
        canvas, box_top = self._draw_story_box(canvas, story_text)
        cv2.putText(
            canvas,
            f"{emo.upper()}  ({conf:.2f})",
            (20, box_top - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        status_y = web_y + web_h + 26
        status = "ACTIVE" if active else "PRESS S TO START"
        cv2.putText(
            canvas, status, (web_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            canvas,
            "S:start  C:calibrate  X:camera  SPACE:next  Q:quit",
            (web_x, status_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        if backend and debug:
            cv2.putText(
                canvas,
                f"detector: {backend}",
                (web_x, status_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
        if not debug:  # study mode: nothing that reveals how the director works
            return canvas
        self._draw_affect_pad(canvas, web_x, status_y + 62, 80, valence, arousal)
        if director_note:
            self._draw_director_note(canvas, director_note)
        if stats:
            cv2.putText(
                canvas,
                stats,
                (web_x, status_y + 62 + 80 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
        return canvas
