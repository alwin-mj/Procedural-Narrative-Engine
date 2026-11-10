import cv2
import numpy as np
import pytest

import narrative_engine.eval_cli as cli
from narrative_engine.demo import video_to_gif

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def write_video(path, frames=40, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 6) % 255, dtype=np.uint8)
        cv2.circle(frame, (20 + i * 3, 60), 12, (0, 200, 255), -1)
        w.write(frame)
    w.release()


def test_gif_has_the_requested_size_rate_and_loops_forever(tmp_path):
    video, out = tmp_path / "v.avi", tmp_path / "out" / "demo.gif"
    write_video(video, frames=40, fps=10)
    info = video_to_gif(video, out, fps=5, width=80)
    assert info["frames"] == 20 and info["fps"] == 5.0 and info["width"] == 80
    assert info["seconds"] == 4.0 and 0 < info["size_mb"] < 1
    with Image.open(out) as gif:
        assert gif.n_frames == 20 and gif.size == (80, 60) and gif.info["loop"] == 0
        assert gif.info["duration"] == 200  # 5 frames per second


def test_gif_respects_max_seconds_and_never_enlarges(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, frames=40, fps=10)
    short = video_to_gif(video, tmp_path / "a.gif", fps=10, width=400, max_seconds=1.5)
    assert short["frames"] == 15 and short["width"] == 160  # source was narrower than 400


def test_gif_errors_are_clear(tmp_path):
    with pytest.raises(ValueError, match="cannot open"):
        video_to_gif(tmp_path / "missing.avi", tmp_path / "x.gif")
    empty = tmp_path / "empty.avi"
    cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60)).release()
    with pytest.raises(ValueError):
        video_to_gif(empty, tmp_path / "x.gif")


def test_gif_needs_pillow_and_says_so(tmp_path):
    import sys

    video = tmp_path / "v.avi"
    write_video(video, frames=5)
    saved = sys.modules.get("PIL")
    sys.modules["PIL"] = None  # makes "import PIL" fail
    try:
        with pytest.raises(RuntimeError, match="pip install pillow"):
            video_to_gif(video, tmp_path / "x.gif")
    finally:
        if saved is not None:
            sys.modules["PIL"] = saved


def test_gif_command(tmp_path, capsys):
    video, out = tmp_path / "v.avi", tmp_path / "demo.gif"
    write_video(video)
    assert cli.main(["gif", str(video), "--out", str(out), "--fps", "5", "--width", "80"]) == 0
    assert "Wrote" in capsys.readouterr().out and out.exists()
    assert cli.main(["gif", str(tmp_path / "nope.avi")]) == 1
    assert "recording not found" in capsys.readouterr().err
