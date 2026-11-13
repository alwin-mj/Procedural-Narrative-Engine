"""Command-line entry point, including a real subprocess replay of a recorded video."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import narrative_engine.__main__ as entry

SRC = str(Path(__file__).resolve().parents[1] / "src")


def write_video(path, frames=30, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(frames):
        w.write(np.full((120, 160, 3), (i * 9) % 255, dtype=np.uint8))
    w.release()


class Swap:
    """Temporarily replace entry.App."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self.orig, entry.App = entry.App, self.fake

    def __exit__(self, *exc):
        entry.App = self.orig


def test_invalid_configuration_returns_2_with_a_message(capsys):
    assert entry.main(["--interval", "0"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err
    assert entry.main(["--headless"]) == 2  # headless camera run without --max-frames


def test_bad_story_file_returns_2(tmp_path):
    bad, video = tmp_path / "bad.json", tmp_path / "x.avi"
    bad.write_text(json.dumps({"scenes": []}), encoding="utf-8")
    write_video(video, frames=3)
    assert (
        entry.main(
            [
                "--backend",
                "haar",
                "--story",
                str(bad),
                "--max-frames",
                "3",
                "--headless",
                "--video",
                str(video),
            ]
        )
        == 2
    )


def test_missing_files_are_reported_immediately_with_the_flag_name(tmp_path, capsys):
    cases = [
        (["--video", str(tmp_path / "session.mp4"), "--headless"], "--video: file not found"),
        (["--replay", str(tmp_path / "s.jsonl")], "--replay: file not found"),
        (["--story", str(tmp_path / "s.json")], "--story: story not found"),
        (["--mp-model", str(tmp_path / "m.task")], "--mp-model: file not found"),
        (["--sprites", str(tmp_path / "frames")], "--sprites: folder not found"),
    ]
    for argv, expected in cases:
        assert entry.main(["--backend", "haar", *argv]) == 2
        assert expected in capsys.readouterr().err


def test_missing_input_checks_run_before_the_app_is_built(tmp_path):
    built = []

    class Spy:
        def __init__(self, cfg):
            built.append(cfg)

        def run(self):
            return 0

    with Swap(Spy):
        assert entry.main(["--video", str(tmp_path / "nope.mp4"), "--headless"]) == 2
    assert built == []  # no detector, no model loading for a typo


def test_a_damaged_replay_log_is_a_clean_error_not_a_traceback(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("this is not a telemetry file\n", encoding="utf-8")
    assert entry.main(["--backend", "haar", "--replay", str(bad)]) == 2


def test_keyboard_interrupt_is_a_clean_exit():
    class Interrupted:
        def __init__(self, cfg):
            pass

        def run(self):
            raise KeyboardInterrupt

    with Swap(Interrupted):
        assert entry.main(["--backend", "haar", "--yes"]) == 0


def test_unexpected_error_returns_1_instead_of_a_traceback():
    class Broken:
        def __init__(self, cfg):
            pass

        def run(self):
            raise ZeroDivisionError("bug")

    with Swap(Broken):
        assert entry.main(["--backend", "haar", "--yes"]) == 1


def test_main_passes_the_run_result_through():
    class Fine:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            return 7

    with Swap(Fine):
        assert entry.main(["--backend", "haar", "--yes"]) == 7


def test_headless_replay_through_main_writes_a_report(tmp_path):
    video, out = tmp_path / "in.avi", tmp_path / "report.json"
    write_video(video)
    code = entry.main(
        [
            "--backend",
            "haar",
            "--video",
            str(video),
            "--headless",
            "--report",
            str(out),
            "--no-calibration",
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["frames"] == 30 and data["route"][0] == "wake" and data["backend"] == "none"


def test_cli_subprocess_replay_is_a_working_integration_test(tmp_path):
    """Run ``python -m narrative_engine`` as a user would, on a recorded video."""
    video, out = tmp_path / "in.avi", tmp_path / "report.json"
    write_video(video, frames=40)
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "narrative_engine",
            "--backend",
            "haar",
            "--video",
            str(video),
            "--headless",
            "--report",
            str(out),
            "--no-calibration",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["frames"] == 40 and data["sim_duration_s"] == 4.0
    assert {"route", "ending", "decisions", "metrics", "wall_fps"} <= set(data)
    assert data["metrics"]["detections"] == 40


def test_cli_story_tool_runs_as_a_module():
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, "-m", "narrative_engine.story_tool", "check"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0 and "OK:" in proc.stdout


# ------------------------------------------------------- privacy: consent before the camera
class CountingApp:
    built = []

    def __init__(self, cfg):
        CountingApp.built.append(cfg)

    def run(self):
        return 0


def with_home(tmp_path, fn):
    old = os.environ.get("NARRATIVE_ENGINE_HOME")
    os.environ["NARRATIVE_ENGINE_HOME"] = str(tmp_path / "home")
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("NARRATIVE_ENGINE_HOME", None)
        else:
            os.environ["NARRATIVE_ENGINE_HOME"] = old


def test_the_camera_is_never_opened_without_consent(tmp_path, capsys):
    CountingApp.built.clear()
    with Swap(CountingApp):
        code = with_home(tmp_path, lambda: entry.main(["--backend", "haar"]))
    assert code == 2 and CountingApp.built == []  # no app, no detector, no camera
    assert "PRIVACY NOTICE" in capsys.readouterr().out


def test_yes_accepts_for_one_run_and_remembers_nothing(tmp_path):
    CountingApp.built.clear()
    with Swap(CountingApp):
        assert with_home(tmp_path, lambda: entry.main(["--backend", "haar", "--yes"])) == 0
    assert len(CountingApp.built) == 1
    assert not (tmp_path / "home" / "consent.json").exists()


def test_stored_consent_is_honoured(tmp_path):
    from narrative_engine.consent import record_consent

    CountingApp.built.clear()
    with Swap(CountingApp):

        def run():
            record_consent()
            return entry.main(["--backend", "haar"])

        assert with_home(tmp_path, run) == 0
    assert len(CountingApp.built) == 1


def test_no_consent_is_asked_when_no_camera_is_used(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=5)
    code = with_home(
        tmp_path,
        lambda: entry.main(
            ["--backend", "haar", "--video", str(video), "--headless", "--no-calibration"]
        ),
    )
    assert code == 0  # a recording needs no webcam, so no notice and no prompt


def test_forget_consent_flag_deletes_the_record_and_exits(tmp_path, capsys):
    from narrative_engine.consent import has_consent, record_consent

    def run():
        record_consent()
        assert has_consent()
        code = entry.main(["--forget-consent"])
        return code, has_consent()

    code, still = with_home(tmp_path, run)
    assert code == 0 and still is False
    assert "Consent forgotten" in capsys.readouterr().out
    assert with_home(tmp_path, lambda: entry.main(["--forget-consent"])) == 0
    assert "No stored consent" in capsys.readouterr().out
