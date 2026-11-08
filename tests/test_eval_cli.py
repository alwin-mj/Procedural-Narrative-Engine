"""The evaluation commands, driven by synthetic frames whose pixels encode the expression."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

import narrative_engine.eval_cli as cli
from narrative_engine import study
from narrative_engine import telemetry as tm
from narrative_engine.perception import EmotionDetector
from narrative_engine.perception.detector import Detection
from narrative_engine.perception.mediapipe_backend import FaceObservation

SRC = str(Path(__file__).resolve().parents[1] / "src")
BOX = (10, 10, 60, 60)

# top-left pixel value -> blendshapes the fake MediaPipe "sees"
SHAPES = {
    0: {},  # neutral face
    10: {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9},  # smile
    15: {"mouthSmileLeft": 0.4, "mouthSmileRight": 0.4},  # a face that rests in a slight smile
    20: {"mouthFrownLeft": 0.6, "mouthFrownRight": 0.6, "browDownLeft": 0.5, "browDownRight": 0.5},
    30: {"eyeWideLeft": 0.7, "eyeWideRight": 0.7, "browInnerUp": 0.7, "jawOpen": 0.6},
}
LABEL_PIXEL = {"happy": 10, "uplift": 30, "sad": 20, "neutral": 0}
NO_FACE = 99


class PixelMP:
    def observe(self, rgb):
        v = int(rgb[0, 0, 0])
        if v == NO_FACE:
            return None
        return FaceObservation(BOX, dict(SHAPES[v]))

    def close(self):
        pass


def pixel_frame(v):
    return np.full((80, 100, 3), v, dtype=np.uint8)


def fake_detector(backend, model=None, **kw):
    if backend != "mediapipe":
        return None
    det = EmotionDetector("haar", **kw)
    det._mp = PixelMP()
    return det


class Patched:
    """Temporarily replace cli.make_detector."""

    def __init__(self, fn=fake_detector):
        self.fn = fn

    def __enter__(self):
        self.orig, cli.make_detector = cli.make_detector, self.fn

    def __exit__(self, *exc):
        cli.make_detector = self.orig


def dataset(neutral_pixel=0, takes=3, frames=12, label_pixels=None, groups=(("me", "bright"),)):
    rows, images = [], {}
    pixels = dict(LABEL_PIXEL, neutral=neutral_pixel, **(label_pixels or {}))
    for subject, lighting in groups:
        for label in LABEL_PIXEL:
            for take in range(1, takes + 1):
                for i in range(frames):
                    p = Path(f"{subject}_{lighting}_{label}_{take}_{i}.png")
                    rows.append(
                        {
                            "path": p,
                            "label": label,
                            "take": str(take),
                            "lighting": lighting,
                            "subject": subject,
                        }
                    )
                    images[p] = pixel_frame(pixels[label])
    return rows, images


# ---------------------------------------------------------------- classifier
def test_classifier_scores_a_clean_dataset_perfectly():
    rows, images = dataset()
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images)
    assert res["n_frames"] == 144 and res["detection_rate"] == 1.0
    assert res["accuracy_detected"] == 1.0 and res["take_accuracy"] == 1.0 and res["n_takes"] == 12
    assert [res["confusion"][i][i] for i in range(4)] == [36, 36, 36, 36]
    assert res["calibrated"] is None


def test_classifier_returns_none_for_a_backend_that_is_not_installed():
    rows, images = dataset()
    with Patched():
        assert cli.evaluate_classifier("fer", rows, images) is None


def test_calibration_fixes_a_face_that_rests_in_a_smile_and_uses_identical_test_frames():
    rows, images = dataset(neutral_pixel=15)
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images, calibrate=True)
    plain, cal = res, res["calibrated"]
    assert plain["n_frames"] == cal["n_frames"] == 144 - 12  # the calibration take is held out
    assert cal["accuracy_detected"] > plain["accuracy_detected"]
    assert cal["accuracy_detected"] == 1.0 and plain["accuracy_detected"] < 0.9
    assert plain["per_class"]["neutral"]["recall"] == 0.0  # uncalibrated: neutral read as happy
    assert res["calibration_skipped_groups"] == []


def test_calibration_skips_groups_without_a_neutral_take_and_handles_each_group_separately():
    rows, images = dataset(groups=(("me", "bright"), ("me", "dim")))
    dim_neutral = {r["path"] for r in rows if r["lighting"] == "dim" and r["label"] == "neutral"}
    rows = [r for r in rows if r["path"] not in dim_neutral]
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images, calibrate=True)
    assert res["calibration_skipped_groups"] == [["me", "dim"]]
    assert res["calibrated"]["n_frames"] == 132  # only the bright group could be evaluated


def test_a_too_short_calibration_take_is_skipped_not_silently_used():
    rows, images = dataset(frames=5)  # calibration needs at least 10 face frames
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images, calibrate=True)
    assert res["calibration_skipped_groups"] == [["me", "bright"]]
    assert res["calibrated"]["n_frames"] == 0


def test_frames_without_a_face_are_reported_separately():
    rows, images = dataset(takes=2, frames=10)
    for r in rows[:10]:
        images[r["path"]] = pixel_frame(NO_FACE)
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images)
    assert res["detection_rate"] == pytest.approx(70 / 80)
    assert res["accuracy_all_frames"] < res["accuracy_detected"] == 1.0


def test_labels_file_helpers(tmp_path):
    csv_path = tmp_path / "data" / "labels.csv"
    assert cli.next_take(csv_path, "happy", "me", "dim") == 1
    cli.append_labels(
        csv_path,
        [{"path": "happy/a.png", "label": "happy", "take": 1, "lighting": "dim", "subject": "me"}],
    )
    cli.append_labels(
        csv_path,
        [{"path": "happy/b.png", "label": "happy", "take": 2, "lighting": "dim", "subject": "me"}],
    )
    assert cli.next_take(csv_path, "happy", "me", "dim") == 3
    assert cli.next_take(csv_path, "happy", "me", "bright") == 1  # separate sequence per lighting
    rows = cli.load_labels(csv_path)
    assert [r["path"].name for r in rows] == ["a.png", "b.png"] and rows[0]["take"] == "1"
    header = "path,label,take,lighting,subject\n"
    csv_path.write_text(header + "x.png,angry,1,dim,me\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels.csv:2"):
        cli.load_labels(csv_path)


def test_classifier_command_writes_json_markdown_and_confusion_svg(tmp_path, capsys):
    data = tmp_path / "eval_data"
    rows = []
    for label in LABEL_PIXEL:
        (data / label).mkdir(parents=True)
        for i in range(3):
            name = f"{label}/me_default_t1_{i:02d}.png"
            cv2.imwrite(str(data / name), np.full((90, 120, 3), 128, dtype=np.uint8))
            rows.append(
                {"path": name, "label": label, "take": 1, "lighting": "default", "subject": "me"}
            )
    cli.append_labels(data / "labels.csv", rows)
    out = tmp_path / "out"
    assert (
        cli.main(["classifier", "--data", str(data), "--backends", "haar", "--out", str(out)]) == 0
    )
    text = capsys.readouterr().out
    assert "fewer than 3 takes" in text and "| haar |" in text
    assert "n/a" in text  # no face found, so no accuracy to report
    assert (out / "classifier_haar.json").exists() and (out / "classifier.md").exists()
    assert (out / "confusion_haar.svg").read_text(encoding="utf-8").startswith("<svg")
    saved = json.loads((out / "classifier_haar.json").read_text(encoding="utf-8"))
    assert saved["detection_rate"] == 0.0


def test_classifier_command_reports_unreadable_images(tmp_path, capsys):
    cli.append_labels(
        tmp_path / "labels.csv",
        [{"path": "gone.png", "label": "happy", "take": 1, "lighting": "x", "subject": "me"}],
    )
    assert (
        cli.main(
            [
                "classifier",
                "--data",
                str(tmp_path),
                "--backends",
                "haar",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 1
    )
    assert "cannot read" in capsys.readouterr().err


# ---------------------------------------------------------------- video: flicker + latency
def noisy_timeline(fps=10, seconds=8):
    """Neutral for 4 s then a smile for 4 s, with a one-frame frown glitch every 6th frame."""
    frames, truth = [], []
    for i in range(int(fps * seconds)):
        v = 0 if i < fps * 4 else 10
        frames.append(pixel_frame(20 if i % 6 == 5 else v))
        truth.append("neutral" if i < fps * 4 else "happy")
    return frames


def test_smoothing_reduces_flicker_and_the_cost_is_a_measurable_delay():
    with Patched():
        det = cli.make_detector("mediapipe")
    timeline = [(0.0, 4.0, "neutral"), (4.0, 8.0, "happy")]
    res = cli.evaluate_video(det, noisy_timeline(), 10.0, timeline)
    flips = res["flips_per_minute"]
    assert flips["raw"] > flips["smoothing_only"] >= flips["smoothing_and_hysteresis"]
    assert res["flicker_reduction"] > 0.5
    onset = res["onset_latency_s"]
    assert onset["raw"]["transitions"] == 1 and onset["final"]["missed"] == 0
    assert onset["final"]["median"] >= onset["raw"]["median"]  # smoothing makes it a bit slower
    assert res["accuracy_vs_timeline"]["final"] > res["accuracy_vs_timeline"]["raw"]
    assert res["frames_with_face"] == 80 and res["duration_s"] == 7.9


def test_video_evaluation_without_a_timeline_and_without_faces():
    with Patched():
        det = cli.make_detector("mediapipe")
        res = cli.evaluate_video(det, noisy_timeline(), 10.0)
        assert "flips_per_minute" in res and "onset_latency_s" not in res
        none = cli.evaluate_video(cli.make_detector("mediapipe"), [pixel_frame(NO_FACE)] * 5, 10.0)
    assert "error" in none


def test_timeline_file_is_validated(tmp_path):
    good = tmp_path / "t.csv"
    good.write_text("start_s,end_s,label\n0,4,neutral\n4,8,happy\n", encoding="utf-8")
    assert cli.load_timeline(good) == [(0.0, 4.0, "neutral"), (4.0, 8.0, "happy")]
    good.write_text("start_s,end_s,label\n0,4,furious\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_timeline(good)


def test_video_command(tmp_path, capsys):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80))
    for f in noisy_timeline():
        w.write(f)
    w.release()
    out = tmp_path / "res.json"
    with Patched():
        assert cli.main(["video", str(video), "--backend", "mediapipe", "--out", str(out)]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["backend"] == "mediapipe"
        assert cli.main(["video", str(video), "--backend", "fer"]) == 1  # not installed
    assert cli.main(["video", str(tmp_path / "missing.avi")]) == 1


# ---------------------------------------------------------------- benchmark
def synthetic_frames(n=6):
    return [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(n)]


def test_benchmark_reports_every_stage_with_ordered_percentiles():
    res = cli.benchmark_backend(
        "haar", synthetic_frames(), iterations=10, capture_ms=[1.0, 2.0, 3.0]
    )
    for stage in ("face", "inference", "smoothing", "render", "total"):
        s = res["stages_ms"][stage]
        assert 0 <= s["p50"] <= s["p95"]
    assert res["detector_fps_single_thread"] > 0 and res["install_size_mb"] >= 0
    assert res["capture_ms"]["p50"] == 2.0 and "threaded" not in res
    assert res["cpu_percent_of_one_core"] > 0


def test_benchmark_threaded_measurement_reports_camera_to_result_latency():
    res = cli.benchmark_backend("haar", synthetic_frames(), iterations=4, threaded_s=0.8)
    t = res["threaded"]
    assert t["detector_hz"] > 0 and t["camera_to_result_ms_p95"] >= t["camera_to_result_ms_p50"] > 0


def test_unavailable_backend_is_skipped():
    with Patched(lambda backend, model=None, **kw: None):
        assert cli.benchmark_backend("haar", synthetic_frames(), 3) is None


def test_benchmark_command_in_process_and_isolated(tmp_path, capsys):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for f in synthetic_frames(8):
        w.write(f)
    w.release()
    common = [
        "benchmark",
        "--video",
        str(video),
        "--backends",
        "haar",
        "--frames",
        "6",
        "--iterations",
        "6",
        "--threaded-seconds",
        "0",
    ]
    out = tmp_path / "bench"
    assert cli.main(common + ["--no-isolate", "--out", str(out)]) == 0
    data = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert data["results"][0]["backend"] == "haar" and (out / "benchmark.md").exists()
    assert "| haar |" in capsys.readouterr().out

    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = SRC + (os.pathsep + old if old else "")
    try:  # one subprocess per backend, so memory figures are not mixed up
        assert cli.main(common + ["--out", str(tmp_path / "iso")]) == 0
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old
    iso = json.loads((tmp_path / "iso" / "benchmark.json").read_text(encoding="utf-8"))
    assert iso["results"][0]["backend"] == "haar"


def test_benchmark_command_reports_a_missing_video(tmp_path, capsys):
    assert cli.main(["benchmark", "--video", str(tmp_path / "x.avi"), "--backends", "haar"]) == 1
    assert "video file not found" in capsys.readouterr().err


def test_small_helpers():
    assert cli.installed_size_mb(["numpy"]) > 1.0
    assert cli.installed_size_mb(["definitely-not-a-package-xyz"]) == 0.0
    rss = cli.peak_rss_mb()
    assert rss is None or rss > 10
    assert cli.pct(0.5) == "50.0%" and cli.pct(None) == "n/a" and cli.ms(1.234) == "1.2"
    assert cli.md_table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


# ---------------------------------------------------------------- LLM timing
class StreamHandler(BaseHTTPRequestHandler):
    pieces = ["Arin ", "walks ", "into the busy ", "market, grinning."]
    gap = 0.02

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        assert json.loads(self.rfile.read(length))["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for i, piece in enumerate(self.pieces):
            time.sleep(self.gap)
            line = json.dumps({"response": piece, "done": i == len(self.pieces) - 1})
            self.wfile.write((line + "\n").encode())
            self.wfile.flush()

    def log_message(self, *a):
        pass


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_stream_generate_measures_time_to_first_token_and_total():
    server, host = serve(StreamHandler)
    try:
        text, ttft, total = cli.stream_generate(host, "tiny", "prompt", timeout_s=5)
    finally:
        server.shutdown()
    assert text == "Arin walks into the busy market, grinning."
    assert 0.015 <= ttft < total and total >= 0.07


def test_llm_benchmark_counts_good_junk_and_failed_requests():
    server, host = serve(StreamHandler)
    try:
        good = cli.benchmark_llm("tiny", host, beats=4, timeout_s=5)

        class Junk(StreamHandler):
            pieces = ["I'm sorry, ", "I cannot ", "help with that."]

        server2, host2 = serve(Junk)
        try:
            junk = cli.benchmark_llm("tiny", host2, beats=4, timeout_s=5)
        finally:
            server2.shutdown()
    finally:
        server.shutdown()
    assert good["fallback_rate"] == 0.0 and good["ttft_ms"]["n"] == 4
    assert good["total_ms"]["median"] > 60
    assert junk["fallback_rate"] == 1.0 and junk["rejected_by_checks"] == 4 and junk["errors"] == 0
    assert list(junk["rejection_reasons"]) == ["refusal instead of narration"]
    down = cli.benchmark_llm("tiny", "http://127.0.0.1:9", beats=3, timeout_s=1)
    assert down["errors"] == 3 and down["fallback_rate"] == 1.0
    assert (
        cli.main(
            [
                "llm",
                "--model",
                "tiny",
                "--host",
                "http://127.0.0.1:9",
                "--beats",
                "2",
                "--timeout",
                "1",
            ]
        )
        == 1
    )


# ---------------------------------------------------------------- counterfactual + study CLI
def make_session(tmp_path, seconds=20, hz=10):
    w = tm.TelemetryWriter(tmp_path / "logs", {"participant": "P01", "mode": "adaptive"})
    for i in range(seconds * hz):
        t = i / hz
        v = 0.6 if int(t) % 8 < 4 else -0.4
        mood = "happy" if v > 0 else "sad"
        s = {"happy": 0.0, "uplift": 0.0, "sad": 0.0, "neutral": 0.0}
        s[mood] = 1.0
        w.write("tick", t, **tm.tick_record(Detection(mood, 0.9, (50, 40, 80, 80), s, v, 0.3)))
    w.close()
    return w.path


def test_counterfactual_replays_one_session_under_each_mode(tmp_path, capsys):
    path = make_session(tmp_path)
    rows = cli.counterfactual(path, ["adaptive", "fixed", "random"], seed=2, interval_s=2.0)
    assert [r["mode"] for r in rows] == ["adaptive", "fixed", "random"]
    assert all(r["route"][0] == "wake" and r["scenes"] >= 3 for r in rows)
    assert rows[1]["route"] != rows[0]["route"]
    assert rows == cli.counterfactual(path, ["adaptive", "fixed", "random"], 2, 2.0)  # reproducible
    assert cli.main(["counterfactual", str(path), "--interval", "2", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "adaptive" in out and "fixed" in out and "random" in out and "not re-generated" in out


def test_study_plan_and_analysis_commands_end_to_end(tmp_path, capsys):
    out = tmp_path / "study"
    assert cli.main(["study-plan", "--participants", "6", "--seed", "3", "--out", str(out)]) == 0
    plan_text = (out / "plan.csv").read_text(encoding="utf-8")
    assert "narrative-engine" in plan_text
    assert cli.main(["study-analyze", str(out / "responses.csv")]) == 1  # blank sheet: nothing yet
    assert "no completed responses" in capsys.readouterr().err

    rows = []
    for i in range(6):
        for cond, val in (("fixed", 2), ("random", 3), ("adaptive", 5)):
            rows.append(
                {
                    "participant": f"P{i + 1:02d}",
                    "condition": cond,
                    "run": 1,
                    **{q: val for q in study.QUESTIONS},
                    "comment": "",
                }
            )
    import csv as _csv

    with open(out / "responses.csv", "w", newline="", encoding="utf-8") as f:
        wr = _csv.DictWriter(f, fieldnames=study.RESPONSE_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    res = tmp_path / "res"
    assert cli.main(["study-analyze", str(out / "responses.csv"), "--out", str(res)]) == 0
    assert "adaptive vs fixed" in capsys.readouterr().out
    assert (res / "study_results.json").exists() and (res / "study_results.md").exists()
    assert (res / "study_responsive.svg").read_text(encoding="utf-8").startswith("<svg")


def test_parser_requires_a_known_command():
    with pytest.raises(SystemExit):
        cli.main([])
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


# ---------------------------------------------------------------- fail fast on a missing file
def test_missing_inputs_fail_fast_before_any_detector_is_built(tmp_path, capsys):
    built = []

    def spy(backend, model=None, **kw):
        built.append(backend)
        return None

    cases = [
        (
            ["video", str(tmp_path / "clip2.mp4"), "--timeline", str(tmp_path / "t.csv")],
            "video file not found",
        ),
        (
            ["classifier", "--data", str(tmp_path / "eval_data"), "--backends", "haar"],
            "labels file",
        ),
        (
            ["benchmark", "--video", str(tmp_path / "clip.mp4"), "--backends", "haar"],
            "video file not found",
        ),
        (["counterfactual", str(tmp_path / "s.jsonl")], "telemetry log not found"),
        (["study-analyze", str(tmp_path / "responses.csv")], "response sheet not found"),
    ]
    with Patched(spy):
        for argv, expected in cases:
            assert cli.main(argv) == 1
            assert expected in capsys.readouterr().err
    assert built == []  # nothing heavy was loaded just to report a typo


def test_a_missing_timeline_is_reported_even_when_the_video_exists(tmp_path, capsys):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80))
    for f in noisy_timeline():
        w.write(f)
    w.release()
    with Patched():
        assert cli.main(["video", str(video), "--timeline", str(tmp_path / "t.csv")]) == 1
    assert "timeline file not found" in capsys.readouterr().err


def test_detector_fallback_warning_is_silenced_for_evaluation_commands(tmp_path):
    import logging

    cli.main(["study-plan", "--participants", "6", "--out", str(tmp_path)])
    detector_log = logging.getLogger("narrative_engine.perception.detector")
    assert detector_log.level == logging.ERROR


def test_classifier_results_carry_confidence_intervals_and_the_table_shows_them():
    rows, images = dataset(takes=3, frames=12)
    for r in rows[::5]:  # make some predictions wrong so the intervals are not degenerate
        images[r["path"]] = pixel_frame(30)
    with Patched():
        res = cli.evaluate_classifier("mediapipe", rows, images)
    lo, hi = res["accuracy_ci95_over_takes"]
    assert 0 <= lo <= res["accuracy_detected"] <= hi <= 1
    assert res["take_hits"] <= res["n_takes"] == 12
    tlo, thi = res["take_accuracy_ci95"]
    assert 0 <= tlo <= thi <= 1
    row = cli.summary_row("mediapipe", res)
    assert "[" in row[3] and "-" in row[3] and "/" in row[6] and row[7] == 12


def test_summary_formatting_helpers():
    assert cli.pct_ci(0.495, [0.2, 0.79]) == "49.5% [20-79%]"
    assert cli.pct_ci(0.5, None) == "50.0%" and cli.pct_ci(None, [0, 1]) == "n/a"
    one_take = {
        "n_takes": 1,
        "take_hits": 1,
        "take_accuracy_ci95": [0.2, 1.0],
        "n_frames": 5,
        "detection_rate": 1.0,
        "accuracy_detected": 1.0,
        "accuracy_ci95_over_takes": None,
        "balanced_accuracy": 1.0,
        "macro_f1": 1.0,
    }
    row = cli.summary_row("x", one_take)
    assert row[3] == "100.0%" and row[6].startswith("1/1 [20-100%]")  # no interval with 1 take
