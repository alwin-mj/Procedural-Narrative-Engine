"""Real-time plumbing: frame sources, capture + perception threads, metrics, degradation.

Threads (when ``Config.threaded`` is on)::

    FrameGrabber ──latest frame──▶ PerceptionWorker ──latest result──▶ UI thread
    (camera pace)                  (throttled, downscaled,             (renders ~30 FPS)
                                    drops old frames)

Each hand-off is a single "latest value" slot, not a queue. If the detector is slower than the
camera, old frames are simply skipped: the newest data always wins, and nothing piles up.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np

from .perception.detector import Box, Detection

log = logging.getLogger(__name__)

Frame = np.ndarray
Clock = Callable[[], float]


# ---------------------------------------------------------------------- frame sources
class CameraSource:
    """A webcam via OpenCV. ``read`` returns mirrored frames."""

    realtime = True
    fps = 30.0

    def __init__(self, index: int = 0, mirror: bool = True) -> None:
        self.index = index
        self.mirror = mirror
        self.finished = False
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.release()
        self._cap = cv2.VideoCapture(self.index)
        return bool(self._cap.isOpened())

    def read(self) -> Tuple[bool, Optional[Frame]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, cv2.flip(frame, 1) if self.mirror else frame

    def reopen(self) -> bool:
        return self.open()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource:
    """A video file, for replay, tests and benchmarks. Optionally paced to real time."""

    def __init__(self, path: Union[str, Path], realtime: bool = False) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.fps = 30.0
        self.finished = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._next_t = 0.0

    def open(self) -> bool:
        self.release()
        self.finished = False
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            return False
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._next_t = 0.0
        return True

    def read(self) -> Tuple[bool, Optional[Frame]]:
        if self._cap is None or self.finished:
            return False, None
        if self.realtime:
            now = time.monotonic()
            if self._next_t == 0.0:
                self._next_t = now
            if now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t += 1.0 / self.fps
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.finished = True
            return False, None
        return True, frame

    def reopen(self) -> bool:
        return False  # a file that ended is not "disconnected"

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ReconnectingSource:
    """Wraps a source: counts read failures, reports "lost", and retries opening it.

    ``read`` returns a frame or ``None``; it never raises.
    """

    def __init__(
        self,
        source: Any,
        lost_after: int = 30,
        retry_s: float = 2.0,
        clock: Clock = time.monotonic,
    ) -> None:
        self.source = source
        self.lost_after = lost_after
        self.retry_s = retry_s
        self._clock = clock
        self.connected = True
        self.ended = False
        self._fails = 0
        self._last_try = 0.0

    def read(self) -> Optional[Frame]:
        try:
            ok, frame = self.source.read()
        except Exception as exc:  # a flaky driver must not take the app down
            log.debug("source read raised: %s", exc)
            ok, frame = False, None
        if ok:
            if not self.connected:
                log.info("Camera reconnected")
            self._fails, self.connected = 0, True
            return frame
        if getattr(self.source, "finished", False):
            self.ended = True
            return None
        self._fails += 1
        if self._fails >= self.lost_after and self.connected:
            self.connected = False
            self._last_try = self._clock()
            log.warning("Camera stopped delivering frames; will keep trying to reconnect")
        if not self.connected and self._clock() - self._last_try >= self.retry_s:
            self._last_try = self._clock()
            try:
                if self.source.reopen():
                    self._fails, self.connected = 0, True
                    log.info("Camera reopened")
            except Exception as exc:
                log.debug("reopen failed: %s", exc)
        return None

    def release(self) -> None:
        try:
            self.source.release()
        except Exception as exc:
            log.debug("release failed: %s", exc)


# ---------------------------------------------------------------------- metrics
def percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


class Metrics:
    """Rolling timing statistics (frame rate, detector rate, latency percentiles)."""

    def __init__(self, window: int = 300, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._frame_ts: Deque[float] = deque(maxlen=window)
        self._det_ts: Deque[float] = deque(maxlen=window)
        self.perception_ms: Deque[float] = deque(maxlen=window)  # detector inference time
        self.latency_ms: Deque[float] = deque(maxlen=window)  # camera frame -> result ready
        self.frames = 0
        self.detections = 0

    def record_frame(self) -> None:
        self._frame_ts.append(self._clock())
        self.frames += 1

    def record_perception(self, inference_ms: float, latency_ms: float) -> None:
        self._det_ts.append(self._clock())
        self.perception_ms.append(inference_ms)
        self.latency_ms.append(latency_ms)
        self.detections += 1

    @staticmethod
    def _rate(ts: Deque[float]) -> float:
        pts = list(ts)[-60:]
        if len(pts) < 2 or pts[-1] <= pts[0]:
            return 0.0
        return (len(pts) - 1) / (pts[-1] - pts[0])

    def fps(self) -> float:
        return self._rate(self._frame_ts)

    def detector_hz(self) -> float:
        return self._rate(self._det_ts)

    def snapshot(self) -> Dict[str, float]:
        p = list(self.perception_ms)
        lat = list(self.latency_ms)
        return {
            "frames": self.frames,
            "detections": self.detections,
            "fps": round(self.fps(), 1),
            "detector_hz": round(self.detector_hz(), 1),
            "inference_ms_p50": round(percentile(p, 50), 2),
            "inference_ms_p95": round(percentile(p, 95), 2),
            "latency_ms_p50": round(percentile(lat, 50), 2),
            "latency_ms_p95": round(percentile(lat, 95), 2),
        }

    def hud_text(self) -> str:
        p95 = percentile(list(self.perception_ms), 95)
        return f"{self.fps():.0f} fps | detector {self.detector_hz():.0f} Hz | p95 {p95:.0f} ms"


# ---------------------------------------------------------------------- perception
class PerceptionResult(NamedTuple):
    detection: Detection
    captured_at: float  # when the camera frame was grabbed
    finished_at: float  # when the detector finished with it
    frame_id: int


def downscale(frame: Frame, max_width: int) -> Tuple[Frame, float]:
    """Shrink to ``max_width`` (keeping aspect). Returns the frame and the scale factor used."""
    w = frame.shape[1]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0
    scale = max_width / w
    small = cv2.resize(
        frame, (max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    return small, scale


def rescale_box(box: Optional[Box], factor: float) -> Optional[Box]:
    if box is None or factor == 1.0:
        return box
    x, y, w, h = box
    return (
        int(round(x * factor)),
        int(round(y * factor)),
        int(round(w * factor)),
        int(round(h * factor)),
    )


def degrade(
    det: Detection, age_s: float, stale_after_s: float, drop_after_s: float, half_life_s: float
) -> Tuple[Detection, bool]:
    """Graceful degradation when perception falls behind.

    Fresh result: unchanged. Older than ``stale_after_s``: keep the last stable mood but let
    valence/arousal decay toward neutral (halving every ``half_life_s``). Older than
    ``drop_after_s``: also drop the face box, so the story treats the viewer as gone.
    Returns ``(detection, lagging)``.
    """
    if age_s <= stale_after_s:
        return det, False
    k = 0.5 ** ((age_s - stale_after_s) / max(1e-6, half_life_s))
    box = det.box if age_s < drop_after_s else None
    return det._replace(box=box, valence=det.valence * k, arousal=det.arousal * k), True


class FrameGrabber:
    """Background thread that keeps the newest camera frame available."""

    def __init__(self, source: ReconnectingSource, clock: Clock = time.monotonic) -> None:
        self.source = source
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._id = 0
        self._captured_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> Tuple[Optional[Frame], int, float]:
        with self._lock:
            return self._frame, self._id, self._captured_at

    @property
    def connected(self) -> bool:
        return self.source.connected

    @property
    def ended(self) -> bool:
        return self.source.ended

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read()
            if frame is None:
                if self.source.ended:
                    return
                self._stop.wait(0.01)  # failed read: do not spin
                continue
            with self._lock:
                self._frame, self._captured_at = frame, self._clock()
                self._id += 1


class PerceptionWorker:
    """Runs the detector on the newest frame, at most ``hz`` times per second, off-thread."""

    def __init__(
        self,
        grabber: FrameGrabber,
        detector: Any,
        hz: float = 15.0,
        max_width: int = 640,
        metrics: Optional[Metrics] = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.grabber = grabber
        self.detector = detector
        self.interval = 1.0 / hz if hz > 0 else 0.0
        self.max_width = max_width
        self.metrics = metrics
        self._clock = clock
        self._lock = threading.Lock()
        self._result: Optional[PerceptionResult] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self.processed = 0
        self.errors = 0

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> Optional[PerceptionResult]:
        with self._lock:
            return self._result

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        last_id = 0
        last_start = -1e9
        while not self._stop.is_set():
            wait = self.interval - (self._clock() - last_start)
            if wait > 0 and self._stop.wait(wait):
                break
            frame, fid, captured_at = self.grabber.latest()  # re-read: skip anything older
            if frame is None or fid == last_id:
                self._stop.wait(0.005)
                continue
            last_id, last_start = fid, self._clock()
            small, scale = downscale(frame, self.max_width)
            t0 = time.perf_counter()
            try:
                det = self.detector.detect_stable(small)
            except Exception as exc:  # one bad frame must not kill perception
                self.errors += 1
                if self.errors <= 3 or self.errors % 100 == 0:
                    log.warning("perception error #%d: %s", self.errors, exc)
                continue
            inference_ms = (time.perf_counter() - t0) * 1000.0
            det = det._replace(box=rescale_box(det.box, 1.0 / scale))
            done = self._clock()
            with self._lock:
                self._result = PerceptionResult(det, captured_at, done, fid)
            self.processed += 1
            if self.metrics is not None:
                self.metrics.record_perception(inference_ms, (done - captured_at) * 1000.0)


class Pipeline:
    """Camera thread + perception thread, started and stopped together."""

    def __init__(
        self,
        source: Any,
        detector: Any,
        *,
        hz: float = 15.0,
        max_width: int = 640,
        lost_after: int = 30,
        retry_s: float = 2.0,
        metrics: Optional[Metrics] = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.reader = ReconnectingSource(source, lost_after, retry_s, clock)
        self.grabber = FrameGrabber(self.reader, clock)
        self.worker = PerceptionWorker(self.grabber, detector, hz, max_width, metrics, clock)

    def start(self) -> None:
        self.grabber.start()
        self.worker.start()

    def frame(self) -> Optional[Frame]:
        return self.grabber.latest()[0]

    def result(self) -> Optional[PerceptionResult]:
        return self.worker.latest()

    @property
    def camera_ok(self) -> bool:
        return self.grabber.connected

    @property
    def ended(self) -> bool:
        return self.grabber.ended

    def stop(self) -> bool:
        """Stop both threads and release the source. Returns True if both exited cleanly."""
        worker_ok = self.worker.stop()
        grabber_ok = self.grabber.stop()
        if grabber_ok:
            self.reader.release()  # unsafe to release while a read may be in flight
        else:
            log.warning("camera thread did not stop in time; leaving the device to the OS")
        return worker_ok and grabber_ok
