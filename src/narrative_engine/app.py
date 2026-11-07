"""Main loop: camera -> perception -> director -> renderer."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Set

import cv2
import numpy as np

from . import __version__
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
from .director.realizer import BeatRequest, LLMRealizer, RealizerWorker, check_ollama
from .perception import Detection, EmotionDetector, FaceMotion, MotionDetector
from .pipeline import (
    CameraSource,
    Metrics,
    Pipeline,
    ReconnectingSource,
    VideoFileSource,
    degrade,
)
from .render import GameWindow, PlayerAnimation
from .telemetry import ReplayDetector, ReplaySource, TelemetryWriter, read_session, tick_record

log = logging.getLogger(__name__)

PROMPT_TEXT = "Press S to start the adventure!"
NOTICE_SECONDS = 2.0
KEY_QUIT = ord("q")
KEY_START = ord("s")
KEY_NEXT = ord(" ")
KEY_CALIBRATE = ord("c")
STARTUP_TIMEOUT_S = 10.0
NO_FACE_TEXT = "I can't see you. Move into view to continue."
CAMERA_LOST_TEXT = "Camera disconnected. Reconnecting..."
_SYNC = object()  # step(): "run the detector yourself, on this thread"
_NO_FACE = Detection(
    "neutral", 0.0, None, {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 1.0}
)


class SimClock:
    """Clock driven by the video's frame index, so replays run faster than real time."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


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
        self._rng = rng or random.Random(cfg.seed)
        self._replay_ticks = None
        if cfg.replay is not None and detector is None:
            session = read_session(cfg.replay)
            self._replay_ticks = session.ticks
            detector = ReplayDetector(self._replay_ticks)
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
            ok, message = check_ollama(cfg.llm_host, cfg.llm)
            if ok:
                log.info("LLM: %s", message)
            else:
                log.warning("LLM: %s. The authored text will be used instead.", message)
            llm = LLMRealizer(cfg.llm, cfg.llm_host, cfg.llm_timeout_s)
            self.worker = RealizerWorker(llm, clock=clock)
        self._requested: Set[Any] = set()
        self._recent: Deque[str] = deque(maxlen=3)

        self.metrics = Metrics()
        self.telemetry: Optional[TelemetryWriter] = None
        if cfg.telemetry is not None:
            self.telemetry = TelemetryWriter(cfg.telemetry, self._telemetry_meta())
        self._t0: Optional[float] = None
        self._last_tick_t = -1e9
        self._started_once = False
        self._paused_reason: Optional[str] = None
        self._telemetry_closed = False
        self.history: Deque[str] = deque(maxlen=cfg.history_len)
        self.active = False
        self.last_decision: Optional[Decision] = None
        self.decision_log: List[str] = []
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

        # latest perception, so rendering can repeat it between detector updates
        self._det: Detection = _NO_FACE
        self._walking = False
        self._lagging = False
        self._last_result_id = -1
        self._no_face_since: Optional[float] = None
        self._seen_window = False
        self._closed = False

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

    # ------------------------------------------------------------------ telemetry
    def _telemetry_meta(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "version": __version__,
            "participant": cfg.participant,
            "mode": cfg.director_mode,
            "backend_requested": cfg.backend,
            "story": self.graph.title,
            "seed": cfg.seed,
            "interval_s": cfg.advance_interval_s,
            "mood_bias": cfg.mood_bias,
            "llm": cfg.llm,
            "study": cfg.study,
            "replay_of": str(cfg.replay) if cfg.replay else None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _tel(self, kind: str, now: float, **data: Any) -> None:
        if self.telemetry is None:
            return
        if self._t0 is None:
            self._t0 = now
        self.telemetry.write(kind, now - self._t0, **data)

    def _tel_tick(self, det: Detection, now: float) -> None:
        if self.telemetry is None:
            return
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        min_gap = 1.0 / self.cfg.telemetry_hz if self.cfg.telemetry_hz > 0 else 0.0
        if t - self._last_tick_t + 1e-9 < min_gap:
            return
        self._last_tick_t = t
        self.telemetry.write("tick", t, **tick_record(det))

    def _close_telemetry(self, now: float) -> None:
        """Write the closing record once and release the file."""
        if self.telemetry is None or self._telemetry_closed:
            return
        self._telemetry_closed = True
        s = self.session
        self._tel(
            "session_end",
            now,
            route=list(s.state.visited),
            moods=list(s.state.moods),
            finished=s.finished,
            ending=s.ending_id,
            metrics=self.metrics.snapshot(),
            llm=self.worker.stats() if self.worker is not None else None,
        )
        self.telemetry.close()

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
        if len(self.decision_log) < 500:
            self.decision_log.append(decision.reason)
        self._last_chosen = mood
        self._last_scores = dict(self._latest_scores)
        self._last_advance = now
        self._acc_v = self._acc_a = 0.0
        self._acc_n = 0
        log.info("director: %s", decision.reason)
        self._tel(
            "scene",
            now,
            id=s.current_id,
            frm=decision.from_id,
            to=decision.to_id,
            mood=mood,
            state=decision.state,
            target=round(decision.target, 3),
            desired=round(decision.desired, 3),
            v=round(affect.valence, 3),
            a=round(affect.arousal, 3),
            reason=decision.reason,
        )
        if s.finished:
            log.info("ending: %s", s.ending_id)
            self._tel("ending", now, id=s.ending_id)
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
        self._tel("restart" if self._started_once else "start", now)
        self._started_once = True
        self._tel(
            "scene",
            now,
            id=self.session.current_id,
            frm=None,
            to=self.session.current_id,
            mood="neutral",
            reason="start",
        )
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
        self._tel("calibration", now, ok=bool(ok))
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
            self._tel("reaction", now, mood=emo)

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

    # ------------------------------------------------------------------ perception
    def _perceive(self, frame: np.ndarray, result: Any, now: float) -> "tuple[Detection, bool]":
        """The detection to act on this frame, and whether it is a *new* reading.

        ``result`` is ``_SYNC`` (run the detector here), ``None`` (the worker has produced
        nothing yet) or a :class:`PerceptionResult` from the worker thread.
        """
        if result is _SYNC:
            t0 = time.perf_counter()
            det = self.detector.detect_stable(frame)
            ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.record_perception(ms, ms)
            self._lagging = False
            return det, True
        if result is None:
            return _NO_FACE, False
        cfg = self.cfg
        det, self._lagging = degrade(
            result.detection,
            now - result.finished_at,
            cfg.stale_after_s,
            cfg.stale_drop_s,
            cfg.stale_half_life_s,
        )
        fresh = result.frame_id != self._last_result_id
        self._last_result_id = result.frame_id
        return det, fresh

    # ------------------------------------------------------------------ one frame
    def step(
        self,
        frame: np.ndarray,
        result: Any = _SYNC,
        camera_ok: bool = True,
        render: bool = True,
    ) -> Optional[np.ndarray]:
        """Process one frame and (optionally) return the composed output image.

        With the default ``result`` the detector runs on this thread. In threaded mode the
        caller passes the worker's latest :class:`PerceptionResult` (or ``None``), so this
        method never waits for inference.
        """
        cfg = self.cfg
        now = self._clock()
        det, fresh = self._perceive(frame, result, now)
        emo, conf, box, scores = det.emotion, det.confidence, det.box, det.scores
        self._det = det

        # State that should advance once per detector reading, not once per rendered frame.
        if fresh:
            self._walking = (
                self.motion.update(box) if cfg.motion_mode == "face" else self.motion.update(frame)
            )
            self._latest_scores, self._latest_emo = scores, emo
            self._latest_affect = Affect(det.valence, det.arousal)
            self.history.append(emo)
            self._tel_tick(det, now)

        face_seen = box is not None and camera_ok
        if face_seen:
            self._no_face_since = None
        elif self._no_face_since is None:
            self._no_face_since = now
        face_lost = (
            self.active
            and self._no_face_since is not None
            and now - self._no_face_since >= cfg.no_face_pause_s
        )

        calibrating = self._calib_until is not None
        if calibrating and now >= self._calib_until:  # type: ignore[operator]
            self._finish_calibration(now)
            calibrating = False

        reason = "camera" if not camera_ok else "face" if face_lost else None
        if self.active and reason != self._paused_reason:
            self._tel("pause" if reason else "resume", now, why=reason or self._paused_reason)
            self._paused_reason = reason

        if not calibrating:
            if self.active and camera_ok and fresh and box is not None:
                self._acc_v += det.valence
                self._acc_a += det.arousal
                self._acc_n += 1
            if self.active and camera_ok and box is not None and conf > cfg.min_confidence:
                if now - self._last_advance > cfg.advance_interval_s:
                    self._advance(now)
            if self.active and (face_lost or not camera_ok):
                self._last_advance = now  # hold the scene timer while the story is paused
            self._track_mood(emo, conf, now)

        if calibrating:
            remaining = math.ceil(self._calib_until - now)  # type: ignore[operator]
            text = f"Hold a neutral face... {remaining}"
        elif not camera_ok:
            text = CAMERA_LOST_TEXT
        elif now < self._notice_until:
            text = self._notice
        elif not self.active:
            text = PROMPT_TEXT
        elif face_lost:
            text = NO_FACE_TEXT
        else:
            text = self._scene_text(emo)
            if now < self._reaction_until and not self.session.finished:
                text = f"A shift: you seem {emo}. {text}"

        self.metrics.record_frame()
        if not render:
            return None
        stats = self.metrics.hud_text() + (" | LAGGING" if self._lagging else "")
        debug = not cfg.study
        return self.window.render(
            frame,
            emo,
            conf,
            box,
            self.active,
            text,
            list(self.history),
            walking=self._walking,
            backend=getattr(self.detector, "active_backend", ""),
            valence=det.valence,
            arousal=det.arousal,
            director_note=self._director_note() if debug else "",
            stats=stats if debug else "",
            debug=debug,
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

    # ------------------------------------------------------------------ running
    def _make_source(self) -> Any:
        cfg = self.cfg
        if self._replay_ticks is not None:
            return ReplaySource(self._replay_ticks)
        if cfg.video:
            return VideoFileSource(cfg.video, realtime=not cfg.headless)
        return CameraSource(cfg.camera_index)

    def _open_error(self) -> str:
        if self.cfg.video:
            return f"Cannot open video {self.cfg.video}"
        return (
            f"Cannot open camera {self.cfg.camera_index}. "
            "Close other apps using it or try --camera 1"
        )

    def _window_closed(self) -> bool:
        """True once the user closed the window with its X button."""
        try:
            visible = cv2.getWindowProperty(self.cfg.window_title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self._seen_window
        if visible >= 1:
            self._seen_window = True
            return False
        return self._seen_window

    def _show(self, out: np.ndarray) -> bool:
        """Display a frame and handle input. Returns False when the app should stop."""
        cv2.imshow(self.cfg.window_title, out)
        key = cv2.waitKey(1) & 0xFF
        return self.handle_key(key) and not self._window_closed()

    def run(self) -> int:
        """Run until quit (window) or until the story/video/frame limit (headless)."""
        source = self._make_source()
        try:
            if self.cfg.headless:
                report = self.run_headless(source)
                self._write_report(report)
                return 0
            if not source.open():
                log.error("%s", self._open_error())
                return 1
            log.info("Running. Keys: S start, C calibrate, SPACE next scene, Q quit.")
            return self._loop_threaded(source) if self.cfg.threaded else self._loop_sync(source)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
        finally:
            self._shutdown()

    def _loop_threaded(self, source: Any) -> int:
        cfg = self.cfg
        pipe = Pipeline(
            source,
            self.detector,
            hz=cfg.perception_hz,
            max_width=cfg.perception_max_width,
            lost_after=cfg.camera_lost_after,
            retry_s=cfg.camera_retry_s,
            metrics=self.metrics,
            clock=self._clock,
        )
        pipe.start()
        interval = 1.0 / cfg.target_fps
        started = time.monotonic()
        try:
            while True:
                t0 = time.perf_counter()
                frame = pipe.frame()
                if frame is None:  # nothing yet, or the stream ended before a frame arrived
                    if pipe.ended:
                        break
                    if time.monotonic() - started > STARTUP_TIMEOUT_S:
                        log.error("No frames received after %.0f s", STARTUP_TIMEOUT_S)
                        return 1
                    time.sleep(0.01)
                    continue
                out = self.step(frame, pipe.result(), pipe.camera_ok)
                if not self._show(out):
                    break
                if pipe.ended:
                    log.info("End of video")
                    break
                spare = interval - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)
        finally:
            if not pipe.stop():
                log.warning("a pipeline thread did not stop cleanly")
            cv2.destroyAllWindows()
        return 0

    def _loop_sync(self, source: Any) -> int:
        cfg = self.cfg
        reader = ReconnectingSource(source, cfg.camera_lost_after, cfg.camera_retry_s, self._clock)
        interval = 1.0 / cfg.target_fps
        frame: Optional[np.ndarray] = None
        try:
            while True:
                t0 = time.perf_counter()
                new = reader.read()
                if reader.ended:
                    log.info("End of video")
                    break
                if new is not None:
                    frame = new
                if frame is None:
                    time.sleep(0.01)
                    continue
                out = self.step(frame, _SYNC, reader.connected)
                if not self._show(out):
                    break
                spare = interval - (time.perf_counter() - t0)
                if spare > 0 and not reader.source.realtime:
                    time.sleep(spare)
        finally:
            reader.release()
            cv2.destroyAllWindows()
        return 0

    def run_headless(self, source: Any) -> Dict[str, Any]:
        """Play a source with no window and return a run report.

        A video file is replayed on a *simulated* clock (frame index / fps), so a 60-second
        recording is processed in seconds and the story still sees 60 seconds go by.
        Single-threaded, so results are reproducible.
        """
        cfg = self.cfg
        if not source.open():
            raise RuntimeError(f"Cannot open {cfg.video or f'camera {cfg.camera_index}'}")
        sim: Optional[SimClock] = None
        if not getattr(source, "realtime", True):
            sim = SimClock()
            self._clock = sim
        fps = float(getattr(source, "fps", 30.0)) or 30.0
        wall0 = time.perf_counter()
        n = 0
        try:
            self._start(self._clock())
            while cfg.max_frames is None or n < cfg.max_frames:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                if sim is not None:
                    recorded = getattr(source, "current_time", None)
                    sim.t = recorded if recorded is not None else n / fps
                self.step(frame, _SYNC, True, render=False)
                n += 1
                if self.session.finished:
                    break
        finally:
            source.release()
        duration = getattr(source, "duration", None)
        report = self._report(
            n, time.perf_counter() - wall0, (duration or n / fps) if sim else None
        )
        self._close_telemetry(self._clock())
        report["telemetry"] = str(self.telemetry.path) if self.telemetry else None
        return report

    def _report(self, frames: int, wall_s: float, sim_s: Optional[float]) -> Dict[str, Any]:
        s = self.session
        return {
            "mode": self.cfg.director_mode,
            "participant": self.cfg.participant,
            "seed": self.cfg.seed,
            "backend": getattr(self.detector, "active_backend", ""),
            "frames": frames,
            "sim_duration_s": round(sim_s, 2) if sim_s is not None else None,
            "wall_s": round(wall_s, 3),
            "wall_fps": round(frames / wall_s, 1) if wall_s > 0 else None,
            "metrics": self.metrics.snapshot(),
            "finished": s.finished,
            "ending": s.ending_id,
            "route": list(s.state.visited),
            "moods": list(s.state.moods),
            "decisions": list(self.decision_log),
            "llm": self.worker.stats() if self.worker is not None else None,
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        log.info(
            "headless run: %d frames in %.2fs, route=%s, ending=%s",
            report["frames"],
            report["wall_s"],
            "->".join(report["route"]),
            report["ending"],
        )
        if self.cfg.report:
            self.cfg.report.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
            log.info("report written to %s", self.cfg.report)

    def _shutdown(self) -> None:
        """Release everything. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._close_telemetry(self._clock())
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()
        if self.worker is not None:
            log.info("LLM stats: %s", self.worker.stats())
            self.worker.close()
