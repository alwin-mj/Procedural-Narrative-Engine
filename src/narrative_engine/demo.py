"""Turn a ``--record`` video into a small GIF for the README."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2


def video_to_gif(
    video: Path,
    out: Path,
    fps: float = 8.0,
    width: int = 560,
    max_seconds: float = 30.0,
    colors: int = 128,
) -> Dict[str, Any]:
    """Convert ``video`` to an optimised looping GIF and report what it produced.

    Frames are thinned to about ``fps``, shrunk to ``width`` pixels and given a shared
    adaptive palette, which keeps a 30-second clip to a few megabytes. Needs Pillow.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("making a GIF needs Pillow: pip install pillow") from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    step = max(1, round(src_fps / fps))
    limit = int(max_seconds * src_fps)
    frames = []
    index = 0
    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            h, w = frame.shape[:2]
            if width and w > width:
                size = (width, round(h * width / w))
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=colors))
        index += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames could be read from {video}")

    effective_fps = src_fps / step
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, round(1000 / effective_fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    return {
        "frames": len(frames),
        "fps": round(effective_fps, 1),
        "width": frames[0].width,
        "seconds": round(len(frames) / effective_fps, 1),
        "size_mb": round(size_mb, 2),
    }
