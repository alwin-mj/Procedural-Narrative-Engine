import time

import cv2
import numpy as np

from narrative_engine.render import GameWindow, PlayerAnimation


def make_sprites(folder):
    cv2.imwrite(str(folder / "a_frame.png"), np.full((20, 30, 4), 255, dtype=np.uint8))  # BGRA
    cv2.imwrite(str(folder / "b_frame.png"), np.full((20, 30, 3), 100, dtype=np.uint8))  # BGR
    cv2.imwrite(str(folder / "c_frame.png"), np.full((20, 30), 50, dtype=np.uint8))  # gray
    (folder / "d_broken.png").write_bytes(b"this is not a png")  # unreadable image
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")  # not an image


def test_frames_are_loaded_sorted_resized_and_converted_to_bgra(tmp_path):
    make_sprites(tmp_path)
    anim = PlayerAnimation(tmp_path, size=(16, 12), fps=10)
    assert anim.n == 3 and anim.has_frames  # broken image and text file skipped
    assert all(f.shape == (12, 16, 4) for f in anim.frames)


def test_missing_folder_means_no_frames_and_no_crash(tmp_path):
    for path in (tmp_path / "nope", None):
        anim = PlayerAnimation(path)
        assert anim.n == 0 and not anim.has_frames and anim.get_frame() is None


def test_frames_advance_at_the_requested_rate(tmp_path):
    make_sprites(tmp_path)
    anim = PlayerAnimation(tmp_path, size=(8, 8), fps=10)
    anim.last_time = time.monotonic()
    first = anim.get_frame()
    assert anim.idx == 0 and first is anim.frames[0]
    anim.last_time -= 1.0  # pretend a second passed
    anim.get_frame()
    assert anim.idx == 1
    for _ in range(3):
        anim.last_time -= 1.0
        anim.get_frame()
    assert anim.idx == 1  # wrapped around after 3 frames: 1 -> 2 -> 0 -> 1


def test_overlay_blends_alpha_clips_and_ignores_offscreen_sprites():
    bg = np.full((50, 50, 3), 200, dtype=np.uint8)
    opaque = np.zeros((10, 10, 4), dtype=np.uint8)
    opaque[..., 3] = 255
    out = PlayerAnimation.overlay_png(bg.copy(), opaque, 5, 5)
    assert out[10, 10].tolist() == [0, 0, 0] and out[0, 0].tolist() == [200, 200, 200]

    clear = np.zeros((10, 10, 4), dtype=np.uint8)  # alpha 0: invisible
    assert (PlayerAnimation.overlay_png(bg.copy(), clear, 5, 5) == bg).all()

    half = np.zeros((10, 10, 4), dtype=np.uint8)
    half[..., 3] = 128
    mixed = PlayerAnimation.overlay_png(bg.copy(), half, 0, 0)[0, 0]
    assert 95 <= int(mixed[0]) <= 105  # about half way between black and 200

    clipped = PlayerAnimation.overlay_png(bg.copy(), opaque, 45, 45)  # hangs off the corner
    assert clipped[49, 49].tolist() == [0, 0, 0] and clipped.shape == bg.shape
    for x, y in ((60, 0), (0, 60), (-20, 0), (0, -20)):
        assert (PlayerAnimation.overlay_png(bg.copy(), opaque, x, y) == bg).all()

    plain = PlayerAnimation.overlay_png(bg.copy(), np.zeros((10, 10, 3), np.uint8), 0, 0)
    assert 15 <= int(plain[0, 0][0]) <= 25  # 3-channel sprite: 90% black + 10% of 200


def test_window_draws_the_sprite_instead_of_the_orb(tmp_path):
    make_sprites(tmp_path)
    anim = PlayerAnimation(tmp_path, size=(40, 40), fps=10)
    with_sprite = GameWindow(320, 240, anim).render(None, "neutral", 0.5, None, True, "hi")
    orb = GameWindow(320, 240, None).render(None, "neutral", 0.5, None, True, "hi")
    assert with_sprite.shape == orb.shape == (240, 320, 3)
    assert not np.array_equal(with_sprite[120:160, 140:180], orb[120:160, 140:180])


def test_window_particles_wrap_around_the_edges():
    win = GameWindow(320, 240)
    for i, p in enumerate(win.particles[:4]):
        p["x"], p["y"] = [-50.0, 400.0, 100.0, 100.0][i], [10.0, 10.0, -5.0, 300.0][i]
        p["vx"] = p["vy"] = 0.0
    win.render(None, "uplift", 0.5, None, True, "x")
    assert win.particles[0]["x"] == 330 and win.particles[1]["x"] == -10
    assert win.particles[2]["y"] == 120 and win.particles[3]["y"] == 0
