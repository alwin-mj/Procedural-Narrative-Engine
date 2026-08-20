import cv2
import numpy as np
import math
import time

class GameWindow:
    def __init__(self, width=960, height=540, player_anim=None):
        self.anim_pos = 60
        self.anim_dir = 1
        self.w = width
        self.h = height
        # player_anim is instance of PlayerAnimation or None
        self.player_anim = player_anim

        # particle system for ambience (sparks / fireflies)
        self.particles = []
        self.last_particle_time = time.time()
        for i in range(40):
            self.particles.append({
                "x": np.random.randint(0, self.w),
                "y": np.random.randint(0, self.h//2),
                "vx": np.random.uniform(-0.3, 0.3),
                "vy": np.random.uniform(-0.1, 0.1),
                "r": np.random.randint(1,4),
                "life": np.random.uniform(3.0, 8.0),
                "t": time.time() - np.random.uniform(0,8.0)
            })

    def render(self, frame, emo, conf, box, active, story_text, history=None, walking=False):
        # create gradient background based on emotion
        if emo == "happy":
            top = np.array((60,200,100), dtype=np.uint8)
            bottom = np.array((20,100,40), dtype=np.uint8)
        elif emo == "uplift":
            top = np.array((60,200,255), dtype=np.uint8)
            bottom = np.array((10,60,120), dtype=np.uint8)
        elif emo == "sad":
            top = np.array((120,100,160), dtype=np.uint8)
            bottom = np.array((40,30,60), dtype=np.uint8)
        else:
            top = np.array((140,140,150), dtype=np.uint8)
            bottom = np.array((60,60,70), dtype=np.uint8)

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # vertical gradient
        for y in range(self.h):
            t = y / (self.h - 1)
            color = (top * (1 - t) + bottom * t).astype(np.uint8)
            canvas[y:y+1, :] = color

        # gentle vignette
        yv, xv = np.ogrid[0:self.h, 0:self.w]
        cx, cy = self.w / 2, self.h / 2
        dx = (xv - cx) / cx
        dy = (yv - cy) / cy
        dist = dx*dx + dy*dy
        vignette = 1.0 - np.clip(dist * 0.6, 0.0, 0.8)
        canvas = (canvas.astype(np.float32) * vignette[..., None]).astype(np.uint8)

        # animated particles (fireflies / sparkles)
        now = time.time()
        for p in self.particles:
            age = now - p["t"]
            life_ratio = max(0.0, 1.0 - (age / p["life"]))
            p["x"] += p["vx"]
            p["y"] += p["vy"] - (0.02 if emo == "uplift" else 0.0)
            # wrap particles
            if p["x"] < -10: p["x"] = self.w + 10
            if p["x"] > self.w + 10: p["x"] = -10
            if p["y"] < 0: p["y"] = self.h//2
            if p["y"] > self.h: p["y"] = 0
            alpha = int(180 * life_ratio)
            col = (255, 255, 220) if emo in ("happy","uplift") else (200,200,220)
            cv2.circle(canvas, (int(p["x"]), int(p["y"])), max(1, int(p["r"] + life_ratio*2)), col, -1)

        # player animation or orb
        if self.player_anim:
            frm = self.player_anim.get_frame()
            if frm is not None:
                px = (self.w // 2) - (frm.shape[1] // 2)
                py = (self.h // 2)
                try:
                    canvas = self.player_anim.overlay_png(canvas, frm, px, py)
                except Exception:
                    pass
        else:
            # stylized orb with glow
            self.anim_pos += self.anim_dir * (6 if walking else 3)
            if self.anim_pos > self.w - 80 or self.anim_pos < 80:
                self.anim_dir *= -1
            center = (int(self.anim_pos), self.h // 3)
            for r, a in ((40, 80), (28, 200)):
                overlay = canvas.copy()
                cv2.circle(overlay, center, r, (255,255,255), -1)
                canvas = cv2.addWeighted(overlay, 0.06 if r==40 else 0.12, canvas, 1- (0.06 if r==40 else 0.12), 0)

        # stylized webcam preview: circular mask with border
        # default webcam preview position/size (used for status placement if preview fails)
        web_x, web_y, web_h = 12, 12, 90
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min((self.w // 5) / fw, (self.h // 5) / fh)
            sw, sh = max(90, int(fw * scale)), max(70, int(fh * scale))
            try:
                small = cv2.resize(frame, (sw, sh))
                # circular mask
                mask = np.zeros((sh, sw), dtype=np.uint8)
                cv2.circle(mask, (sw//2, sh//2), min(sw,sh)//2, 255, -1)
                rgb_small = small.copy()
                # prepare region and composite
                x0, y0 = web_x, web_y
                bg_region = canvas[y0:y0+sh, x0:x0+sw]
                # alpha-composite circular area
                for c in range(3):
                    bg_region[:,:,c] = np.where(mask==255, rgb_small[:,:,c], bg_region[:,:,c])
                canvas[y0:y0+sh, x0:x0+sw] = bg_region
                # border
                cv2.circle(canvas, (x0 + sw//2, y0 + sh//2), min(sw,sh)//2 + 3, (255,255,255), 2)
                # draw face box if present (approx)
                if box is not None:
                    x, y, wf, hf = box
                    sx = int((x + wf*0.5) * sw / fw)  # draw center-ish indicator
                    sy = int((y + hf*0.5) * sh / fh)
                    cv2.circle(canvas, (x0 + sx, y0 + sy), 6, (0,255,255), -1)
                # record actual preview height for status placement
                web_h = sh
            except Exception:
                # leave web_h as default if preview fails
                pass

        # Emotion/confidence display (stylized)
        cv2.putText(canvas, f"{emo.upper()}  ({conf:.2f})", (12, self.h - 120),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

        # Draw emotion history bars if provided
        if history:
            from collections import Counter
            cnt = Counter(history)
            keys = ["happy", "uplift", "sad", "neutral"]
            for i, k in enumerate(keys):
                v = cnt.get(k, 0)
                x = self.w - 180
                y = 20 + i*30
                cv2.rectangle(canvas, (x, y), (x + 120, y + 22), (30,30,30), -1)
                cv2.rectangle(canvas, (x+6, y+4), (x+6 + min(110, v*10), y+18), (200,200,200), -1)
                cv2.putText(canvas, f"{k}", (x - 70, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

        # Render story text with rounded semi-transparent box for readability
        lines = self.wrap_text(story_text, 48)
        box_h = 26 * len(lines) + 20
        box_w = self.w - 40
        x0, y0 = 20, self.h - box_h - 20
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (10,10,10), -1)
        # rounded corners by mixing
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x0 + 18, y0 + 22 + i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240,240,240), 2, cv2.LINE_AA)

        # Status instructions: move below webcam preview
        status_x = web_x
        status_y = web_y + web_h + 14
        inst_y = status_y + 26
        status = "ACTIVE" if active else "PRESS S TO START"
        cv2.putText(canvas, status, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(canvas, "S:start  Q:quit  SPACE:next", (status_x, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        return canvas

    def wrap_text(self, text, width):
        if not text:
            return [""]
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if len(line + " " + word) > width:
                lines.append(line)
                line = word
            else:
                if line:
                    line += " "
                line += word
        if line:
            lines.append(line)
        return lines
