"""``narrative-eval``: tools that produce the numbers (and only the numbers) for the write-up.

    narrative-eval collect      record labelled expression frames from your webcam
    narrative-eval classifier   accuracy / confusion matrix per backend on those frames
    narrative-eval video        flicker rate and reaction latency on a recorded video
    narrative-eval benchmark    per-stage latency, FPS, CPU, memory, install size per backend
    narrative-eval llm          time-to-first-token / total time / fallback rate for the LLM
    narrative-eval counterfactual   replay one recorded session under adaptive/fixed/random
    narrative-eval gif          turn a --record video into a README-sized GIF
    narrative-eval study-plan   counterbalanced run order and commands for a user study
    narrative-eval study-analyze    statistics and charts from the response sheet

Nothing here invents data: every command measures what you give it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import MOODS, Config
from .director.graph import load_story
from .director.realizer import BeatRequest, build_prompt, check_output
from .evaluation import (
    classification_report,
    cluster_bootstrap_accuracy,
    confusion_matrix,
    confusion_svg,
    flips_per_minute,
    majority,
    onset_latencies,
    summarize,
    take_level_accuracy,
    wilson_interval,
)
from .perception.detector import MIN_CALIBRATION_SAMPLES, EmotionDetector
from .perception.smoothing import EmotionSmoother
from .pipeline import Metrics, Pipeline
from .study import (
    analyze,
    chart_for,
    load_responses,
    make_plan,
    render_markdown,
    write_plan,
    write_response_template,
)

log = logging.getLogger(__name__)

LABEL_FIELDS = ["path", "label", "take", "lighting", "subject"]
BACKEND_PACKAGES = {
    "haar": ["opencv-python"],
    "mediapipe": ["mediapipe"],
    "fer": ["fer"],
    "deepface": ["deepface"],
}


# ---------------------------------------------------------------------- small helpers
def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pct_ci(x: Optional[float], ci: Optional[Sequence[float]]) -> str:
    """``49.5% [20-79%]``: a percentage with its 95% interval (or just the percentage)."""
    if x is None:
        return "n/a"
    return pct(x) if not ci else f"{pct(x)} [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%]"


def ms(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def require_file(path: Path, what: str) -> None:
    """Fail fast, with a clear message, before any model is loaded."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{what} not found: {path}")


def make_detector(
    backend: str, model: Optional[Path] = None, **kw: Any
) -> Optional[EmotionDetector]:
    """A detector that uses *only* ``backend``, or ``None`` if that backend is not installed."""
    det = EmotionDetector(backend=backend, mediapipe_model=model, **kw)
    return det if backend in det.loaded_backends else None


def installed_size_mb(
    packages: Sequence[str], _seen: Optional[set] = None, depth: int = 0
) -> float:
    """Disk size of a package and its (unconditional) dependency closure, in MB.

    Environment markers and extras are ignored, so this slightly over-counts. It is an
    estimate for comparing backends, not an exact install size.
    """
    seen = _seen if _seen is not None else set()
    total = 0.0
    for name in packages:
        key = name.lower().replace("_", "-")
        if key in seen or depth > 6:
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for f in dist.files or []:
            try:
                total += Path(str(dist.locate_file(f))).stat().st_size
            except OSError:
                pass
        deps = []
        for req in dist.requires or []:
            if "extra ==" in req:
                continue
            m = re.match(r"^([A-Za-z0-9_.-]+)", req)
            if m:
                deps.append(m.group(1))
        total += installed_size_mb(deps, seen, depth + 1) * 1024 * 1024
    return total / (1024 * 1024)


def peak_rss_mb() -> Optional[float]:
    """Peak resident memory of this process (Unix only), in MB."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):  # Windows
        return None
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


# ---------------------------------------------------------------------- collect (interactive)
def append_labels(csv_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def next_take(csv_path: Path, label: str, subject: str, lighting: str) -> int:
    """Next unused take number for this (label, subject, lighting)."""
    if not csv_path.exists():
        return 1
    with open(csv_path, newline="", encoding="utf-8") as f:
        taken = [
            int(r["take"])
            for r in csv.DictReader(f)
            if r["label"] == label and r["subject"] == subject and r["lighting"] == lighting
        ]
    return max(taken, default=0) + 1


def cmd_collect(a: argparse.Namespace) -> int:  # pragma: no cover - needs a camera and a window
    out = Path(a.out)
    csv_path = out / "labels.csv"
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {a.camera}", file=sys.stderr)
        return 1
    plan = [m for _ in range(a.takes_per_label) for m in MOODS]
    print(f"{len(plan)} takes x {a.seconds:g}s at {a.fps:g} fps. Press Q to stop early.")
    try:
        for label in plan:
            take = next_take(csv_path, label, a.subject, a.lighting)
            for count in range(3, 0, -1):  # get ready
                end = time.monotonic() + 1
                while time.monotonic() < end:
                    ok, frame = cap.read()
                    if ok:
                        shown = cv2.flip(frame, 1)
                        cv2.putText(
                            shown,
                            f"Get ready: {label.upper()} in {count}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("collect", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return 0
            rows, t_end, next_shot = [], time.monotonic() + a.seconds, 0.0
            while time.monotonic() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                shown = cv2.flip(frame, 1)
                cv2.putText(
                    shown,
                    f"HOLD: {label.upper()}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("collect", shown)
                if time.monotonic() >= next_shot:
                    next_shot = time.monotonic() + 1.0 / a.fps
                    name = f"{label}/{a.subject}_{a.lighting}_t{take}_{len(rows):02d}.png"
                    (out / label).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / name), shown)  # mirrored, as the app sees it
                    rows.append(
                        {
                            "path": name,
                            "label": label,
                            "take": take,
                            "lighting": a.lighting,
                            "subject": a.subject,
                        }
                    )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0
            append_labels(csv_path, rows)
            print(f"saved {len(rows)} frames: {label} take {take}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------- classifier accuracy
def load_labels(csv_path: Path) -> List[Dict[str, Any]]:
    """Read labels.csv; ``path`` is relative to the CSV's folder."""
    base = csv_path.parent
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for n, r in enumerate(csv.DictReader(f), 2):
            if r.get("label") not in MOODS:
                got = r.get("label")
                raise ValueError(f"{csv_path}:{n}: label must be one of {MOODS}, got {got!r}")
            rows.append(
                {
                    "path": base / r["path"],
                    "label": r["label"],
                    "take": r.get("take") or "1",
                    "lighting": r.get("lighting") or "default",
                    "subject": r.get("subject") or "me",
                }
            )
    return rows


def _score(rows: Sequence[Dict[str, Any]], preds: Sequence[Optional[str]]) -> Dict[str, Any]:
    """Metrics for predictions (``None`` = no face found)."""
    n = len(rows)
    detected = [(r, p) for r, p in zip(rows, preds) if p is not None]
    cm = confusion_matrix([r["label"] for r, _ in detected], [p for _, p in detected])
    rep = classification_report(cm)
    takes: Dict[Tuple[Any, ...], Tuple[str, List[str]]] = {}
    for r, p in zip(rows, preds):
        key = (r["subject"], r["lighting"], r["take"], r["label"])
        takes.setdefault(key, (r["label"], []))
        if p is not None:
            takes[key][1].append(p)
    correct_all = sum(1 for r, p in zip(rows, preds) if p == r["label"])
    keyed = {str(k): v for k, v in takes.items()}
    scored = [(truth, ps) for truth, ps in keyed.values() if ps]
    hits = sum(1 for truth, ps in scored if majority(ps) == truth)
    acc_ci = cluster_bootstrap_accuracy(keyed)
    take_ci = wilson_interval(hits, len(scored)) if scored else None
    return {
        "n_frames": n,
        "detection_rate": len(detected) / n if n else None,
        "accuracy_detected": rep["accuracy"],
        "accuracy_all_frames": correct_all / n if n else None,  # no face counts as wrong
        "balanced_accuracy": rep["balanced_accuracy"],
        "macro_f1": rep["macro_f1"],
        "per_class": rep["per_class"],
        "confusion": cm,
        "accuracy_ci95_over_takes": None if acc_ci is None else [round(x, 4) for x in acc_ci],
        "n_takes": len(takes),
        "take_accuracy": take_level_accuracy(keyed),
        "take_hits": hits,
        "take_accuracy_ci95": None if take_ci is None else [round(x, 4) for x in take_ci],
    }


def _predict(det: EmotionDetector, img: np.ndarray) -> Optional[str]:
    d = det.detect(img)
    return d.emotion if d.box is not None else None


def evaluate_classifier(
    backend: str,
    rows: Sequence[Dict[str, Any]],
    images: Dict[Path, np.ndarray],
    calibrate: bool = False,
    model: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Frame-level accuracy of one backend (``None`` if the backend is not installed)."""
    det = make_detector(backend, model)
    if det is None:
        return None
    result: Dict[str, Any] = {"backend": backend}
    if not (calibrate and det.supports_calibration):
        preds = [_predict(det, images[r["path"]]) for r in rows]
        result.update(_score(rows, preds))
        result["calibrated"] = None
        return result

    # Calibration needs a neutral take per (subject, lighting); those frames are then held out
    # of the evaluation so the calibrated and uncalibrated numbers use identical test frames.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["subject"], r["lighting"])].append(r)
    test_rows: List[Dict[str, Any]] = []
    plain: List[Optional[str]] = []
    calibrated: List[Optional[str]] = []
    skipped = []
    for key, grp in groups.items():
        neutral_takes = sorted({r["take"] for r in grp if r["label"] == "neutral"}, key=str)
        if not neutral_takes:
            skipped.append(key)
            continue
        calib = [r for r in grp if r["label"] == "neutral" and r["take"] == neutral_takes[0]]
        rest = [r for r in grp if r not in calib]
        cdet = make_detector(backend, model)
        assert cdet is not None
        cdet.begin_calibration()
        for r in calib:
            cdet.detect(images[r["path"]])
        if not cdet.finish_calibration():
            skipped.append(key)
            continue
        for r in rest:
            test_rows.append(r)
            plain.append(_predict(det, images[r["path"]]))
            calibrated.append(_predict(cdet, images[r["path"]]))
    result.update(_score(test_rows, plain))
    result["calibrated"] = _score(test_rows, calibrated)
    result["calibration_skipped_groups"] = [list(k) for k in skipped]
    return result


def summary_row(label: str, r: Dict[str, Any]) -> List[Any]:
    """One row of the classifier summary table."""
    hits = f"{r['take_hits']}/{r['n_takes']}" if r["n_takes"] else "n/a"
    take_ci = r["take_accuracy_ci95"]
    take_cell = hits if not take_ci else f"{hits} [{100 * take_ci[0]:.0f}-{100 * take_ci[1]:.0f}%]"
    return [
        label,
        r["n_frames"],
        pct(r["detection_rate"]),
        pct_ci(r["accuracy_detected"], r["accuracy_ci95_over_takes"]),
        pct(r["balanced_accuracy"]),
        pct(r["macro_f1"]),
        take_cell,
        r["n_takes"],
    ]


def cmd_classifier(a: argparse.Namespace) -> int:
    csv_path = Path(a.data) / "labels.csv"
    require_file(csv_path, "labels file (record frames first with `narrative-eval collect`)")
    rows = load_labels(csv_path)
    counts: Dict[str, set] = defaultdict(set)
    for r in rows:
        counts[r["label"]].add((r["subject"], r["lighting"], r["take"]))
    print(
        f"{len(rows)} frames. Takes per label: " + ", ".join(f"{m}={len(counts[m])}" for m in MOODS)
    )
    if any(len(counts[m]) < 3 for m in MOODS):
        print(
            "WARNING: fewer than 3 takes for some label. Frames within a take are "
            "near-duplicates, so treat these numbers as indicative only."
        )
    images: Dict[Path, np.ndarray] = {}
    for r in rows:
        img = cv2.imread(str(r["path"]))
        if img is None:
            print(f"cannot read {r['path']}", file=sys.stderr)
            return 1
        images[r["path"]] = img

    out = Path(a.out)
    summary_rows = []
    for backend in [b.strip() for b in a.backends.split(",") if b.strip()]:
        res = evaluate_classifier(backend, rows, images, a.calibrate, a.mp_model)
        if res is None:
            print(f"- {backend}: not installed, skipped")
            continue
        for group in res.get("calibration_skipped_groups") or []:
            print(
                f"- {backend}: calibration skipped for {tuple(group)}: it needs a neutral take "
                f"with at least {MIN_CALIBRATION_SAMPLES} frames that all show a face"
            )
        save_json(res, out / f"classifier_{backend}.json")
        (out / f"confusion_{backend}.svg").write_text(
            confusion_svg(res["confusion"], MOODS, f"{backend}: confusion (detected frames)"),
            encoding="utf-8",
        )
        summary_rows.append(summary_row(backend, res))
        if res.get("calibrated"):
            summary_rows.append(summary_row(f"{backend} + calibration", res["calibrated"]))
    headers = [
        "backend",
        "frames",
        "face found",
        "accuracy [95% CI over takes]",
        "balanced acc.",
        "macro F1",
        "take-level hits [95% CI]",
        "takes",
    ]
    table = md_table(headers, summary_rows)
    (out / "classifier.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote results, confusion matrices (SVG) and classifier.md to {out}/")
    return 0


# ---------------------------------------------------------------------- video: flicker + latency
def load_timeline(path: Path) -> List[Tuple[float, float, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(float(r["start_s"]), float(r["end_s"]), r["label"]) for r in csv.DictReader(f)]
    for _, _, lab in rows:
        if lab not in MOODS:
            raise ValueError(f"timeline label must be one of {MOODS}, got {lab!r}")
    return rows


def evaluate_video(
    det: EmotionDetector,
    frames: Sequence[np.ndarray],
    fps: float,
    timeline: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> Dict[str, Any]:
    """Flicker (mood changes per minute) with and without smoothing; reaction time vs truth."""
    smoother = EmotionSmoother(det.smoother.buffer_size, det.smoother.window)
    times: List[float] = []
    raw: List[str] = []
    smooth_only: List[str] = []
    final: List[str] = []
    for i, frame in enumerate(frames):
        d = det.detect(frame)
        stable = det.stabilize(d)
        sm = smoother.update(d.scores, d.emotion, min(1.0, max(0.0, d.confidence)))[0]
        if d.box is None:
            continue  # only frames with a face say anything about flicker
        times.append(i / fps)
        raw.append(d.emotion)
        smooth_only.append(sm)
        final.append(stable.emotion)
    out: Dict[str, Any] = {"frames": len(frames), "frames_with_face": len(times), "fps": fps}
    if len(times) < 2:
        out["error"] = "fewer than two frames with a face"
        return out
    dur = times[-1] - times[0]
    streams = {"raw": raw, "smoothing_only": smooth_only, "smoothing_and_hysteresis": final}
    out["duration_s"] = round(dur, 2)
    out["flips_per_minute"] = {k: round(flips_per_minute(v, dur), 2) for k, v in streams.items()}
    base = out["flips_per_minute"]["raw"]
    final_rate = out["flips_per_minute"]["smoothing_and_hysteresis"]
    out["flicker_reduction"] = None if base == 0 else round(1 - final_rate / base, 3)
    if timeline:
        keep, truth = [], []
        for k, t in enumerate(times):
            lab = next((lab for s, e, lab in timeline if s <= t < e), None)
            if lab is not None:
                keep.append(k)
                truth.append(lab)
        if len(truth) >= 2:
            tt = [times[k] for k in keep]
            lat = {
                name: onset_latencies(tt, truth, [seq[k] for k in keep])
                for name, seq in (("raw", raw), ("final", final))
            }
            out["accuracy_vs_timeline"] = {
                name: round(sum(1 for k, tr in zip(keep, truth) if seq[k] == tr) / len(truth), 3)
                for name, seq in (("raw", raw), ("final", final))
            }
            out["onset_latency_s"] = {
                name: {
                    "transitions": len(v),
                    "missed": sum(1 for _, x in v if x is None),
                    **summarize([x for _, x in v if x is not None]),
                }
                for name, v in lat.items()
            }
    return out


def read_video(path: Path, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames: List[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def cmd_video(a: argparse.Namespace) -> int:
    require_file(Path(a.video), "video file")
    if a.timeline:
        require_file(Path(a.timeline), "timeline file")
    det = make_detector(a.backend, a.mp_model)
    if det is None:
        print(f"backend {a.backend!r} is not installed", file=sys.stderr)
        return 1
    frames, fps = read_video(Path(a.video), a.max_frames)
    timeline = load_timeline(Path(a.timeline)) if a.timeline else None
    res = evaluate_video(det, frames, fps, timeline)
    res["backend"] = a.backend
    print(json.dumps(res, indent=2))
    if a.out:
        save_json(res, Path(a.out))
    return 0 if "error" not in res else 1


# ---------------------------------------------------------------------- benchmark
class _LoopSource:
    """Feeds in-memory frames at a fixed rate, like a camera (for the threaded measurement)."""

    realtime = True
    fps = 30.0
    finished = False

    def __init__(self, frames: Sequence[np.ndarray], rate: float = 30.0) -> None:
        self._frames, self._i, self._gap, self._next = frames, 0, 1.0 / rate, 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self._gap
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return True, frame

    def reopen(self) -> bool:
        return True

    def release(self) -> None:
        pass


def benchmark_backend(
    backend: str,
    frames: Sequence[np.ndarray],
    iterations: int,
    model: Optional[Path] = None,
    threaded_s: float = 0.0,
    capture_ms: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Stage latencies, throughput, CPU and memory for one backend."""
    det = make_detector(backend, model)
    if det is None:
        return None
    from .render import GameWindow

    window = GameWindow(960, 540)
    names = ("face", "inference", "smoothing", "render", "total")
    stages: Dict[str, List[float]] = {k: [] for k in names}
    for _ in range(min(5, iterations)):  # warm-up: model load, caches
        det.detect_stable(frames[0])
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for i in range(iterations):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        out = det.detect_stable(frame)
        t1 = time.perf_counter()
        window.render(frame, out.emotion, out.confidence, out.box, True, "benchmark", [out.emotion])
        t2 = time.perf_counter()
        lt = det.last_timings
        stages["face"].append(lt["face_ms"])
        stages["inference"].append(lt["infer_ms"])
        stages["smoothing"].append(lt["smooth_ms"])
        stages["render"].append((t2 - t1) * 1e3)
        stages["total"].append((t1 - t0) * 1e3)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    res: Dict[str, Any] = {
        "backend": backend,
        "iterations": iterations,
        "stages_ms": {
            k: {
                "p50": round(float(np.percentile(v, 50)), 3),
                "p95": round(float(np.percentile(v, 95)), 3),
            }
            for k, v in stages.items()
        },
        "capture_ms": (
            {
                "p50": round(float(np.percentile(capture_ms, 50)), 3),
                "p95": round(float(np.percentile(capture_ms, 95)), 3),
            }
            if capture_ms
            else None
        ),
        "detector_fps_single_thread": round(iterations / wall, 1),
        "cpu_percent_of_one_core": round(100 * cpu / wall, 1) if wall > 0 else None,
        "peak_rss_mb": None if peak_rss_mb() is None else round(peak_rss_mb() or 0.0, 1),
        "install_size_mb": round(installed_size_mb(BACKEND_PACKAGES[backend]), 1),
    }
    if threaded_s > 0:
        metrics = Metrics()
        pipe = Pipeline(_LoopSource(frames), det, hz=15, max_width=640, metrics=metrics)
        pipe.start()
        time.sleep(threaded_s)
        pipe.stop()
        snap = metrics.snapshot()
        res["threaded"] = {
            "detector_hz": snap["detector_hz"],
            "camera_to_result_ms_p50": snap["latency_ms_p50"],
            "camera_to_result_ms_p95": snap["latency_ms_p95"],
            "note": "add up to one render interval (about 33 ms at 30 FPS) for time to screen",
        }
    return res


def _frames_for_benchmark(a: argparse.Namespace) -> Tuple[List[np.ndarray], List[float]]:
    """Frames to benchmark on, and per-frame source read times in ms."""
    reads: List[float] = []
    frames: List[np.ndarray] = []
    if a.video:
        cap = cv2.VideoCapture(str(a.video))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {a.video}")
        while len(frames) < a.frames:
            t0 = time.perf_counter()
            ok, f = cap.read()
            if not ok:
                break
            reads.append((time.perf_counter() - t0) * 1e3)
            frames.append(f)
        cap.release()
    else:  # pragma: no cover - needs a camera
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            raise ValueError(f"cannot open camera {a.camera}")
        while len(frames) < a.frames:
            t0 = time.perf_counter()
            ok, f = cap.read()
            if ok:
                reads.append((time.perf_counter() - t0) * 1e3)
                frames.append(cv2.flip(f, 1))
        cap.release()
    if not frames:
        raise ValueError("no frames to benchmark on")
    return frames, reads


def cmd_benchmark(a: argparse.Namespace) -> int:
    if a.video:
        require_file(a.video, "video file")
    try:
        frames, reads = _frames_for_benchmark(a)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    results = []
    for backend in [b.strip() for b in a.backends.split(",") if b.strip()]:
        if a.isolate:  # one process per backend, so memory numbers are not contaminated
            cmd = [
                sys.executable,
                "-m",
                "narrative_engine.eval_cli",
                "benchmark",
                "--no-isolate",
                "--backends",
                backend,
                "--frames",
                str(a.frames),
                "--iterations",
                str(a.iterations),
                "--threaded-seconds",
                str(a.threaded_seconds),
                "--json-only",
            ]
            cmd += ["--video", str(a.video)] if a.video else ["--camera", str(a.camera)]
            if a.mp_model:
                cmd += ["--mp-model", str(a.mp_model)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                tail = proc.stderr.strip()[-300:]
                print(f"- {backend}: benchmark failed: {tail}", file=sys.stderr)
                continue
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            results.extend(payload["results"])
            continue
        res = benchmark_backend(
            backend, frames, a.iterations, a.mp_model, a.threaded_seconds, reads
        )
        if res is None:
            print(f"- {backend}: not installed, skipped", file=sys.stderr)
            continue
        results.append(res)

    if a.json_only:
        print(json.dumps({"results": results}))
        return 0
    rows = []
    for r in results:
        s = r["stages_ms"]
        rows.append(
            [
                r["backend"],
                ms(s["face"]["p50"]),
                ms(s["inference"]["p50"]),
                ms(s["inference"]["p95"]),
                ms(s["smoothing"]["p50"]),
                ms(s["render"]["p50"]),
                ms(s["total"]["p95"]),
                r["detector_fps_single_thread"],
                r["cpu_percent_of_one_core"],
                r["peak_rss_mb"],
                r["install_size_mb"],
            ]
        )
    headers = [
        "backend",
        "face p50 ms",
        "infer p50",
        "infer p95",
        "smooth p50",
        "render p50",
        "detect+smooth p95",
        "detector FPS",
        "CPU % (1 core)",
        "peak RSS MB",
        "install MB",
    ]
    table = md_table(headers, rows)
    print(table)
    if a.out:
        out = Path(a.out)
        save_json({"frames": len(frames), "results": results}, out / "benchmark.json")
        (out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------- LLM timing
def stream_generate(
    host: str, model: str, prompt: str, timeout_s: float, opener: Any = urllib.request.urlopen
) -> Tuple[str, Optional[float], float]:
    """Run one streaming Ollama request. Returns (text, time-to-first-token s, total s)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.7, "num_predict": 100},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first: Optional[float] = None
    parts: List[str] = []
    with opener(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("response", "")
            if piece and first is None:
                first = time.perf_counter() - t0
            parts.append(piece)
            if chunk.get("done"):
                break
    return "".join(parts), first, time.perf_counter() - t0


def benchmark_llm(
    model: str, host: str, beats: int, timeout_s: float, story_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Time-to-first-token, total time and fallback rate over the story's own beats."""
    graph = load_story(story_path)
    work: List[Tuple[str, str]] = []
    for node in graph.nodes.values():
        for mood in MOODS:
            work.append((node.variants[mood], mood))
    work = work[:beats]
    ttfts: List[float] = []
    totals: List[float] = []
    rejected = errors = 0
    reasons: Dict[str, int] = defaultdict(int)
    for text, mood in work:
        req = BeatRequest(
            authored=text, mood=mood, title=graph.title, protagonist=graph.protagonist
        )
        try:
            out, ttft, total = stream_generate(host, model, build_prompt(req), timeout_s)
        except (OSError, ValueError) as exc:
            errors += 1
            log.debug("llm request failed: %s", exc)
            continue
        if ttft is not None:
            ttfts.append(ttft * 1e3)
        totals.append(total * 1e3)
        _, why = check_output(out, graph.protagonist, text)
        if why is not None:
            rejected += 1
            reasons[why] += 1
    n = len(work)
    return {
        "model": model,
        "beats": n,
        "errors": errors,
        "rejected_by_checks": rejected,
        "rejection_reasons": dict(reasons),
        "fallback_rate": round((errors + rejected) / n, 3) if n else None,
        "ttft_ms": summarize(ttfts),
        "total_ms": summarize(totals),
    }


def cmd_llm(a: argparse.Namespace) -> int:
    res = benchmark_llm(a.model, a.host, a.beats, a.timeout)
    print(json.dumps(res, indent=2))
    if a.out:
        save_json(res, Path(a.out))
    return 0 if res["errors"] < res["beats"] else 1


# ---------------------------------------------------------------------- counterfactual replay
def counterfactual(
    session_path: Path, modes: Sequence[str], seed: int, interval_s: float
) -> List[Dict[str, Any]]:
    """Replay one recorded session under several director policies."""
    from .app import App

    rows = []
    for mode in modes:
        cfg = Config(
            backend="haar",
            headless=True,
            replay=session_path,
            director_mode=mode,
            seed=seed,
            advance_interval_s=interval_s,
            calibrate_on_start=False,
        )
        app = App(cfg)
        rep = app.run_headless(app._make_source())
        rows.append(
            {
                "mode": mode,
                "route": rep["route"],
                "scenes": len(rep["route"]),
                "finished": rep["finished"],
                "ending": rep["ending"],
                "moods": rep["moods"],
            }
        )
    return rows


def cmd_counterfactual(a: argparse.Namespace) -> int:
    require_file(Path(a.session), "telemetry log")
    modes = [m.strip() for m in a.modes.split(",")]
    rows = counterfactual(Path(a.session), modes, a.seed, a.interval)
    print(
        md_table(
            ["mode", "scenes", "finished", "ending", "route"],
            [
                [r["mode"], r["scenes"], r["finished"], r["ending"], " > ".join(r["route"])]
                for r in rows
            ],
        )
    )
    print(
        "\nNote: the recording stops when the original session ended, so a policy that takes a "
        "longer route may not finish. Viewer reactions are replayed, not re-generated: the "
        "viewer cannot respond to the different story."
    )
    return 0


# ---------------------------------------------------------------------- user study
def cmd_study_plan(a: argparse.Namespace) -> int:
    plan = make_plan(a.participants, a.seed, a.interval, a.telemetry_dir)
    out = Path(a.out)
    write_plan(plan, out / "plan.csv")
    write_response_template(plan, out / "responses.csv")
    table_rows = [[r["participant"], r["run"], r["condition"]] for r in plan]
    print(md_table(["participant", "run", "condition"], table_rows))
    print(
        f"\nWrote {out / 'plan.csv'} (with the exact command for every run) and a blank "
        f"{out / 'responses.csv'}."
    )
    return 0


def cmd_study_analyze(a: argparse.Namespace) -> int:
    require_file(Path(a.responses), "response sheet")
    rows = load_responses(Path(a.responses))
    result = analyze(rows, seed=a.seed)
    md = render_markdown(result)
    print(md)
    if a.out:
        out = Path(a.out)
        save_json(result, out / "study_results.json")
        (out / "study_results.md").write_text(md + "\n", encoding="utf-8")
        for q in result["questions"]:
            svg = chart_for(result, q)
            if svg:
                (out / f"study_{q}.svg").write_text(svg, encoding="utf-8")
        print(f"\nWrote results, tables and charts to {out}/")
    return 0


# ---------------------------------------------------------------------- demo GIF
def cmd_gif(a: argparse.Namespace) -> int:
    from .demo import video_to_gif

    require_file(Path(a.video), "recording")
    res = video_to_gif(Path(a.video), Path(a.out), a.fps, a.width, a.max_seconds)
    print(
        f"Wrote {a.out}: {res['frames']} frames, {res['seconds']} s, {res['width']} px wide, "
        f"{res['size_mb']} MB"
    )
    if res["size_mb"] > 10:
        print("That is large for a README. Try --width 480, --fps 6 or a shorter --max-seconds.")
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="narrative-eval",
        description="Measure the system: benchmarks, classifier accuracy, user-study analysis.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="record labelled expression frames from the webcam")
    c.add_argument("--out", default="eval_data")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--subject", default="me")
    c.add_argument("--lighting", default="default", help="e.g. bright, dim, backlit")
    c.add_argument("--takes-per-label", type=int, default=2)
    c.add_argument("--seconds", type=float, default=3.0)
    c.add_argument("--fps", type=float, default=5.0)
    c.set_defaults(func=cmd_collect)

    c = sub.add_parser("classifier", help="accuracy and confusion matrix per backend")
    c.add_argument("--data", default="eval_data", help="folder containing labels.csv")
    c.add_argument("--backends", default="haar,mediapipe,fer,deepface")
    c.add_argument(
        "--calibrate", action="store_true", help="also evaluate with per-group calibration"
    )
    c.add_argument("--mp-model", type=Path, default=None)
    c.add_argument("--out", default="docs/eval")
    c.set_defaults(func=cmd_classifier)

    c = sub.add_parser("video", help="flicker and reaction latency on a recorded video")
    c.add_argument("video")
    c.add_argument("--backend", default="mediapipe")
    c.add_argument("--timeline", default=None, help="CSV: start_s,end_s,label")
    c.add_argument("--max-frames", type=int, default=None)
    c.add_argument("--mp-model", type=Path, default=None)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_video)

    c = sub.add_parser("benchmark", help="per-stage latency, FPS, CPU, memory, install size")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--camera", type=int)
    c.add_argument("--backends", default="haar,mediapipe,fer,deepface")
    c.add_argument("--frames", type=int, default=150, help="frames to load")
    c.add_argument("--iterations", type=int, default=300)
    c.add_argument(
        "--threaded-seconds",
        type=float,
        default=3.0,
        help="also measure camera-to-result latency in the threaded pipeline (0 = skip)",
    )
    c.add_argument("--isolate", dest="isolate", action="store_true", default=True)
    c.add_argument("--no-isolate", dest="isolate", action="store_false")
    c.add_argument("--json-only", action="store_true", help=argparse.SUPPRESS)
    c.add_argument("--mp-model", type=Path, default=None)
    c.add_argument("--out", default="docs/eval")
    c.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("llm", help="time-to-first-token, total time, fallback rate (Ollama)")
    c.add_argument("--model", required=True)
    c.add_argument("--host", default="http://localhost:11434")
    c.add_argument("--beats", type=int, default=12)
    c.add_argument("--timeout", type=float, default=30.0)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_llm)

    c = sub.add_parser("counterfactual", help="replay a session under different director modes")
    c.add_argument("session")
    c.add_argument("--modes", default="adaptive,fixed,random")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--interval", type=float, default=3.5)
    c.set_defaults(func=cmd_counterfactual)

    c = sub.add_parser("gif", help="turn a --record video into a README-sized GIF")
    c.add_argument("video", help="a video made with narrative-engine --record")
    c.add_argument("--out", default="docs/demo.gif")
    c.add_argument("--fps", type=float, default=8.0)
    c.add_argument("--width", type=int, default=560)
    c.add_argument("--max-seconds", type=float, default=30.0)
    c.set_defaults(func=cmd_gif)

    c = sub.add_parser("study-plan", help="counterbalanced run order and commands")
    c.add_argument("--participants", type=int, required=True)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--interval", type=float, default=5.0)
    c.add_argument("--telemetry-dir", default="logs/study")
    c.add_argument("--out", default="study")
    c.set_defaults(func=cmd_study_plan)

    c = sub.add_parser("study-analyze", help="statistics and charts from the response sheet")
    c.add_argument("responses")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="docs/eval")
    c.set_defaults(func=cmd_study_analyze)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Asking for a backend that is not installed is normal here (it is just skipped), so the
    # detector's "falling back to Haar" warning would only be noise.
    logging.getLogger("narrative_engine.perception.detector").setLevel(logging.ERROR)
    try:
        return int(args.func(args))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
