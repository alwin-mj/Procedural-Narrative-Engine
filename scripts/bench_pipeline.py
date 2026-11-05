"""Compare render frame rate with and without the threaded pipeline.

Uses a synthetic camera and a detector with an artificial delay, so you can see the effect of a
slow detector without a webcam:

    python scripts/bench_pipeline.py --delay-ms 60 --seconds 3

Single loop: every rendered frame waits for the detector, so FPS collapses to ~1000/delay.
Threaded: rendering stays near the target FPS and the detector just runs at its own pace.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from narrative_engine.app import _SYNC, App
from narrative_engine.config import Config
from narrative_engine.perception.detector import Detection
from narrative_engine.pipeline import Pipeline


class SyntheticCamera:
    realtime = True
    fps = 30.0
    finished = False

    def read(self):
        time.sleep(1 / 30)  # a 30 FPS camera
        return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def reopen(self):
        return True

    def release(self):
        pass


class SlowDetector:
    supports_calibration = False
    calibrated = False
    active_backend = "synthetic"

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def detect_stable(self, frame):
        time.sleep(self.delay_s)
        scores = {"happy": 1.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
        return Detection("happy", 0.9, (200, 120, 200, 220), scores, 0.5, 0.2)


def run_single(delay_s: float, seconds: float) -> dict:
    app = App(Config(backend="haar", calibrate_on_start=False), detector=SlowDetector(delay_s))
    cam = SyntheticCamera()
    end, frames = time.monotonic() + seconds, 0
    while time.monotonic() < end:
        _, frame = cam.read()
        app.step(frame, _SYNC)
        frames += 1
    return {"mode": "single loop", "render_fps": frames / seconds,
            "detector_hz": app.metrics.snapshot()["detector_hz"]}


def run_threaded(delay_s: float, seconds: float) -> dict:
    cfg = Config(backend="haar", calibrate_on_start=False)
    app = App(cfg, detector=SlowDetector(delay_s))
    pipe = Pipeline(SyntheticCamera(), app.detector, hz=cfg.perception_hz,
                    max_width=cfg.perception_max_width, metrics=app.metrics)
    pipe.start()
    while pipe.frame() is None:
        time.sleep(0.01)
    interval = 1 / cfg.target_fps
    end, frames = time.monotonic() + seconds, 0
    while time.monotonic() < end:
        t0 = time.perf_counter()
        app.step(pipe.frame(), pipe.result(), pipe.camera_ok)
        frames += 1
        time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
    pipe.stop()
    return {"mode": "threaded", "render_fps": frames / seconds,
            "detector_hz": app.metrics.snapshot()["detector_hz"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--delay-ms", type=float, default=60.0, help="artificial detector latency")
    p.add_argument("--seconds", type=float, default=3.0)
    a = p.parse_args()
    print(f"detector latency {a.delay_ms:.0f} ms, {a.seconds:.0f} s per mode")
    for fn in (run_single, run_threaded):
        r = fn(a.delay_ms / 1000.0, a.seconds)
        line = f"  {r['mode']:12s} render {r['render_fps']:5.1f} FPS"
        print(line + f" | detector {r['detector_hz']:5.1f} Hz")


if __name__ == "__main__":
    main()
