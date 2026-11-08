import random
import xml.etree.ElementTree as ET

import pytest

from narrative_engine import evaluation as ev

L = ("happy", "uplift", "sad", "neutral")


# ---------------------------------------------------------------- classifier metrics
def test_confusion_matrix_counts_rows_as_truth_and_columns_as_prediction():
    truth = ["happy", "happy", "sad", "neutral"]
    cm = ev.confusion_matrix(truth, ["happy", "sad", "sad", "neutral"])
    assert cm == [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_confusion_matrix_rejects_bad_input():
    with pytest.raises(ValueError):
        ev.confusion_matrix(["happy"], ["happy", "sad"])
    with pytest.raises(ValueError):
        ev.confusion_matrix(["happy"], ["angry"])


def test_report_matches_hand_computed_values():
    cm = [[8, 0, 2, 0], [0, 5, 0, 0], [1, 0, 3, 0], [0, 0, 0, 0]]
    rep = ev.classification_report(cm, L)
    assert rep["n"] == 19 and rep["accuracy"] == pytest.approx(16 / 19)
    happy = rep["per_class"]["happy"]
    assert happy["precision"] == pytest.approx(8 / 9) and happy["recall"] == pytest.approx(0.8)
    assert happy["f1"] == pytest.approx(2 * (8 / 9) * 0.8 / (8 / 9 + 0.8))
    empty = rep["per_class"]["neutral"]  # a class with no samples at all
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] is None
    assert rep["balanced_accuracy"] == pytest.approx((0.8 + 1.0 + 0.75) / 3)  # skips empty class


def test_report_handles_no_data_and_zero_f1():
    assert ev.classification_report([[0] * 4 for _ in range(4)])["accuracy"] is None
    cm = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert ev.classification_report(cm)["per_class"]["happy"]["f1"] == 0.0


def test_majority_and_take_level_accuracy():
    assert ev.majority(["a", "b", "b"]) == "b"
    assert ev.majority(["a", "b"]) == "a"  # tie: first seen
    with pytest.raises(ValueError):
        ev.majority([])
    takes = {
        "t1": ("happy", ["happy", "happy", "sad"]),
        "t2": ("sad", ["happy", "happy", "sad"]),
        "t3": ("neutral", []),
    }
    assert ev.take_level_accuracy(takes) == pytest.approx(1 / 3)
    assert ev.take_level_accuracy({}) is None


# ---------------------------------------------------------------- flicker and latency
def test_flip_counting_and_rate():
    assert ev.count_flips(["a", "a", "b", "a", "a"]) == 2
    assert ev.count_flips([]) == 0 and ev.count_flips(["x"]) == 0
    assert ev.flips_per_minute(["a", "b", "a", "b"], 30.0) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        ev.flips_per_minute(["a"], 0)


def test_onset_latency_per_transition_and_misses():
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    truth = ["n", "n", "h", "h", "h", "s", "s", "s"]
    pred = ["n", "n", "n", "n", "h", "n", "n", "n"]  # happy found 2 s late; sad never found
    assert ev.onset_latencies(t, truth, pred) == [("h", 2.0), ("s", None)]
    assert ev.onset_latencies(t, ["n"] * 8, pred) == []  # no transitions
    with pytest.raises(ValueError):
        ev.onset_latencies([0, 1], ["a"], ["a"])


# ---------------------------------------------------------------- statistics
def test_summarize():
    s = ev.summarize([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["mean"] == 3 and s["median"] == 3 and s["min"] == 1 and s["max"] == 5
    assert s["sd"] == pytest.approx(1.5811, abs=1e-4) and s["se"] == pytest.approx(0.7071, abs=1e-4)
    assert ev.summarize([])["n"] == 0 and ev.summarize([])["mean"] is None
    assert ev.summarize([7])["sd"] == 0.0


def test_bootstrap_ci_is_seeded_and_contains_the_mean():
    data = [3, 4, 4, 5, 5, 5, 4, 3]
    lo, hi = ev.bootstrap_ci(data, seed=1)
    assert (lo, hi) == ev.bootstrap_ci(data, seed=1)
    assert lo <= sum(data) / len(data) <= hi and 3 <= lo < hi <= 5
    assert ev.bootstrap_ci([4, 4, 4], seed=0) == (4.0, 4.0)
    with pytest.raises(ValueError):
        ev.bootstrap_ci([])


def test_sign_flip_test_matches_exact_enumeration():
    assert ev.paired_sign_flip_pvalue([1] * 8) == pytest.approx(2 / 256)  # all positive
    assert ev.paired_sign_flip_pvalue([1, -1, 1, -1]) == 1.0  # balanced: nothing to see
    assert ev.paired_sign_flip_pvalue([0, 0, 0]) == 1.0
    assert ev.paired_sign_flip_pvalue([2]) == pytest.approx(1.0)  # one pair: both signs extreme
    assert ev.paired_sign_flip_pvalue([1, 1, 1, 1, 1, -1]) == pytest.approx(14 / 64)
    with pytest.raises(ValueError):
        ev.paired_sign_flip_pvalue([])


def test_sign_flip_monte_carlo_for_large_samples_is_reproducible():
    rng = random.Random(3)
    diffs = [rng.choice([1, 1, 1, 2, -1]) for _ in range(30)]
    p1 = ev.paired_sign_flip_pvalue(diffs, seed=5)
    assert p1 == ev.paired_sign_flip_pvalue(diffs, seed=5) and 0 < p1 < 0.05
    assert ev.paired_sign_flip_pvalue([0.5] * 30, seed=5) < 0.001


def test_effect_size_and_holm():
    assert ev.cohens_dz([1, 2, 3]) == pytest.approx(2.0)
    assert ev.cohens_dz([1]) is None and ev.cohens_dz([2, 2, 2]) is None  # constant, non-zero
    assert ev.cohens_dz([0, 0]) == 0.0
    assert ev.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ev.holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # never decreasing
    assert ev.holm_adjust([]) == []


# ---------------------------------------------------------------- charts
def test_confusion_svg_is_well_formed_and_escapes_text():
    cm = [[5, 1, 0, 0], [0, 4, 1, 0], [0, 0, 6, 0], [1, 0, 0, 3]]
    svg = ev.confusion_svg(cm, L, "A <b> & C")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "&lt;b&gt;" in svg and svg.count("<rect") == 1 + 16
    for lab in L:
        assert lab in svg


def test_bar_svg_is_well_formed_for_normal_and_extreme_values():
    groups = [
        {"label": "fixed", "mean": 2.4, "lo": 1.9, "hi": 2.9},
        {"label": "adaptive", "mean": 4.9, "lo": 4.6, "hi": 5.0},
        {"label": "low", "mean": 1.0, "lo": 1.0, "hi": 1.2},
    ]
    root = ET.fromstring(ev.bar_svg("Responsive", groups))
    assert root.tag.endswith("svg")
    assert "4.90" in ev.bar_svg("t", groups) and "adaptive" in ev.bar_svg("t", groups)
    assert ET.fromstring(ev.bar_svg("wide scale", groups[:1], ymin=0, ymax=100)) is not None


# ---------------------------------------------- honest uncertainty for small samples
def test_wilson_interval_matches_known_values():
    lo, hi = ev.wilson_interval(3, 7)
    assert lo == pytest.approx(0.158, abs=0.002) and hi == pytest.approx(0.750, abs=0.002)
    lo, hi = ev.wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.434, abs=0.002)
    lo, hi = ev.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.722, abs=0.002)
    with pytest.raises(ValueError):
        ev.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        ev.wilson_interval(5, 4)


def test_cluster_bootstrap_resamples_takes_not_frames():
    perfect = {f"t{i}": ("happy", ["happy"] * 10) for i in range(4)}
    assert ev.cluster_bootstrap_accuracy(perfect) == (1.0, 1.0)

    mixed = {"good": ("happy", ["happy"] * 15), "bad": ("sad", ["happy"] * 15)}  # 50% overall
    lo, hi = ev.cluster_bootstrap_accuracy(mixed, seed=3)
    assert lo == 0.0 and hi == 1.0  # two takes cannot pin down accuracy: the interval says so
    assert (lo, hi) == ev.cluster_bootstrap_accuracy(mixed, seed=3)  # seeded: repeatable

    seven = {}  # 4 of 7 takes right, like a small real dataset
    for i in range(7):
        seven[f"t{i}"] = ("happy", ["happy"] * 12 if i < 4 else ["sad"] * 12)
    lo, hi = ev.cluster_bootstrap_accuracy(seven, seed=1)
    assert lo <= 4 / 7 <= hi and hi - lo > 0.4  # wide, as 7 takes deserve


def test_cluster_bootstrap_needs_at_least_two_takes_with_frames():
    assert ev.cluster_bootstrap_accuracy({}) is None
    assert ev.cluster_bootstrap_accuracy({"only": ("happy", ["happy"] * 5)}) is None
    two_with_one_empty = {"a": ("happy", ["happy"] * 5), "b": ("sad", [])}
    assert ev.cluster_bootstrap_accuracy(two_with_one_empty) is None
