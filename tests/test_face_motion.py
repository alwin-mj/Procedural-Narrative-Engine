from narrative_engine.perception.motion import FaceMotion


def run(m, boxes):
    return [m.update(b) for b in boxes]


def test_still_face_is_not_walking():
    m = FaceMotion(threshold=0.03)
    assert not any(run(m, [(100, 100, 80, 80)] * 20))


def test_small_jitter_is_ignored():
    m = FaceMotion(threshold=0.03)
    boxes = [(100 + (i % 2), 100, 80, 80) for i in range(30)]  # 1 px jitter on 80 px face
    assert not any(run(m, boxes))


def test_sustained_movement_is_walking():
    m = FaceMotion(threshold=0.03)
    boxes = [(100 + i * 6, 100, 80, 80) for i in range(15)]  # 6 px/frame = 7.5% of width
    assert run(m, boxes)[-1] is True


def test_single_fast_frame_does_not_trigger_median_filter():
    m = FaceMotion(threshold=0.03)
    boxes = [(100, 100, 80, 80)] * 8 + [(180, 100, 80, 80)] + [(180, 100, 80, 80)] * 4
    out = run(m, boxes)
    assert not any(out)


def test_losing_the_face_resets_state():
    m = FaceMotion(threshold=0.03)
    run(m, [(100 + i * 6, 100, 80, 80) for i in range(10)])
    assert m.update(None) is False and m.last_motion == 0.0
    assert m.update((300, 100, 80, 80)) is False
