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
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scenes": []}), encoding="utf-8")
    assert entry.main(["--backend", "haar", "--story", str(bad), "--max-frames", "3",
                       "--headless", "--video", str(tmp_path / "x.avi")]) == 2


def test_keyboard_interrupt_is_a_clean_exit():
    class Interrupted:
        def __init__(self, cfg):
            pass

        def run(self):
            raise KeyboardInterrupt

    with Swap(Interrupted):
        assert entry.main(["--backend", "haar"]) == 0


def test_unexpected_error_returns_1_instead_of_a_traceback():
    class Broken:
        def __init__(self, cfg):
            pass

        def run(self):
            raise ZeroDivisionError("bug")

    with Swap(Broken):
        assert entry.main(["--backend", "haar"]) == 1


def test_main_passes_the_run_result_through():
    class Fine:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            return 7

    with Swap(Fine):
        assert entry.main(["--backend", "haar"]) == 7


def test_headless_replay_through_main_writes_a_report(tmp_path):
    video, out = tmp_path / "in.avi", tmp_path / "report.json"
    write_video(video)
    code = entry.main(["--backend", "haar", "--video", str(video), "--headless",
                       "--report", str(out), "--no-calibration"])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["frames"] == 30 and data["route"][0] == "wake" and data["backend"] == "none"


def test_cli_subprocess_replay_is_a_working_integration_test(tmp_path):
    """Run ``python -m narrative_engine`` as a user would, on a recorded video."""
    video, out = tmp_path / "in.avi", tmp_path / "report.json"
    write_video(video, frames=40)
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, "-m", "narrative_engine", "--backend", "haar", "--video", str(video),
         "--headless", "--report", str(out), "--no-calibration"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["frames"] == 40 and data["sim_duration_s"] == 4.0
    assert {"route", "ending", "decisions", "metrics", "wall_fps"} <= set(data)
    assert data["metrics"]["detections"] == 40


def test_cli_story_tool_runs_as_a_module():
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-m", "narrative_engine.story_tool", "check"],
                          capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0 and "OK:" in proc.stdout
