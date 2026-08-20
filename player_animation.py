import os
import time
import cv2
import numpy as np

class PlayerAnimation:
    def __init__(self, folder_path, size=(160,160), fps=10):
        self.frames = []
        self.idx = 0
        self.last_time = time.time()
        self.fps = fps
        self.size = size
        if folder_path and os.path.isdir(folder_path):
            for fn in sorted(os.listdir(folder_path)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img = cv2.imread(os.path.join(folder_path, fn), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    img = cv2.resize(img, (self.size[0], self.size[1]), interpolation=cv2.INTER_AREA)
                    # ensure 4 channels for consistent overlay handling
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    if img.shape[2] == 3:
                        b,g,r = cv2.split(img)
                        alpha = np.full(b.shape, 255, dtype=b.dtype)
                        img = cv2.merge([b,g,r,alpha])
                    self.frames.append(img)
        self.n = len(self.frames)

    def get_frame(self):
        if self.n == 0:
            return None
        now = time.time()
        if now - self.last_time >= 1.0 / self.fps:
            self.idx = (self.idx + 1) % self.n
            self.last_time = now
        return self.frames[self.idx]

    @staticmethod
    def overlay_png(bg, fg, x, y):
        # overlay fg (with alpha) onto bg at position (x,y)
        h, w = fg.shape[:2]
        bh, bw = bg.shape[:2]
        if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
            return bg
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bw, x + w)
        y2 = min(bh, y + h)
        fg_x1 = x1 - x
        fg_y1 = y1 - y
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)

        bg_region = bg[y1:y2, x1:x2].astype(float)
        fg_region = fg[fg_y1:fg_y2, fg_x1:fg_x2].astype(float)

        if fg_region.shape[2] == 4:
            alpha = fg_region[:, :, 3:4] / 255.0
            fg_rgb = fg_region[:, :, :3]
            blended = alpha * fg_rgb + (1 - alpha) * bg_region
            bg[y1:y2, x1:x2] = blended.astype(bg.dtype)
        else:
            blended = 0.9 * fg_region[:, :, :3] + 0.1 * bg_region
            bg[y1:y2, x1:x2] = blended.astype(bg.dtype)
        return bg
