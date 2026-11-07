import csv
from collections import Counter

import pytest

from narrative_engine import study as st

QS = list(st.QUESTIONS)


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or st.RESPONSE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def rating_rows(n, adaptive=5, fixed=2, rnd=3, jitter=None):
    rows = []
    for i in range(n):
        for cond, base in (("fixed", fixed), ("random", rnd), ("adaptive", adaptive)):
            val = base + (jitter[i % len(jitter)] if jitter else 0)
            val = max(1, min(5, val))
            rows.append(
                {
                    "participant": f"P{i + 1:02d}",
                    "condition": cond,
                    "run": 1,
                    **{q: val for q in QS},
                    "comment": "",
                }
            )
    return rows


# ---------------------------------------------------------------- plan
def test_plan_is_counterbalanced_across_orders():
    plan = st.make_plan(12, seed=4)
    orders = Counter(
        tuple(r["condition"] for r in plan if r["participant"] == f"P{i:02d}") for i in range(1, 13)
    )
    assert len(orders) == 6 and set(orders.values()) == {2}  # every order exactly twice
    for r in plan:
        assert r["condition"] in st.CONDITIONS and 1 <= r["run"] <= 3
    firsts = Counter(r["condition"] for r in plan if r["run"] == 1)
    assert set(firsts.values()) == {4}  # each condition goes first equally often


def test_plan_commands_are_complete_and_seeds_differ():
    plan = st.make_plan(3, seed=7, interval_s=6, telemetry_dir="logs/x")
    assert len(plan) == 9
    cmd = plan[0]["command"]
    for part in (
        "--study",
        "--mode " + plan[0]["condition"],
        "--participant P01",
        "--interval 6",
        "--telemetry logs/x",
        "--seed",
    ):
        assert part in cmd
    assert len({r["command"] for r in plan}) == 9
    assert plan == st.make_plan(3, seed=7, interval_s=6, telemetry_dir="logs/x")  # reproducible
    assert plan != st.make_plan(3, seed=8, interval_s=6, telemetry_dir="logs/x")


def test_plan_rejects_zero_participants_and_files_roundtrip(tmp_path):
    with pytest.raises(st.StudyError):
        st.make_plan(0)
    plan = st.make_plan(2, seed=1)
    st.write_plan(plan, tmp_path / "a" / "plan.csv")
    st.write_response_template(plan, tmp_path / "a" / "responses.csv")
    rows = list(csv.DictReader(open(tmp_path / "a" / "plan.csv", encoding="utf-8")))
    assert len(rows) == 6 and rows[0]["command"].startswith("narrative-engine")
    assert st.load_responses(tmp_path / "a" / "responses.csv") == []  # blank template = no data yet


# ---------------------------------------------------------------- responses
def test_load_responses_reads_valid_rows(tmp_path):
    write_csv(tmp_path / "r.csv", rating_rows(2))
    rows = st.load_responses(tmp_path / "r.csv")
    assert len(rows) == 6 and rows[0]["responsive"] in (2, 3, 5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].update(responsive=6),
        lambda rows: rows[0].update(responsive=0),
        lambda rows: rows[0].update(engaged="lots"),
        lambda rows: rows[0].update(condition="chaos"),
        lambda rows: rows[0].update(participant=""),
        lambda rows: rows.append(dict(rows[0])),  # duplicate (participant, condition)
    ],
)
def test_load_responses_rejects_bad_rows(tmp_path, mutate):
    rows = rating_rows(2)
    mutate(rows)
    write_csv(tmp_path / "r.csv", rows)
    with pytest.raises(st.StudyError):
        st.load_responses(tmp_path / "r.csv")


def test_missing_file_and_complete_case_selection(tmp_path):
    with pytest.raises(st.StudyError):
        st.load_responses(tmp_path / "nope.csv")
    rows = rating_rows(3)
    rows = [r for r in rows if not (r["participant"] == "P02" and r["condition"] == "random")]
    write_csv(tmp_path / "r.csv", rows)
    loaded = st.load_responses(tmp_path / "r.csv")
    assert st.complete_participants(loaded, "responsive") == ["P01", "P03"]


# ---------------------------------------------------------------- analysis
def test_analysis_finds_a_clear_effect_and_reports_everything(tmp_path):
    write_csv(tmp_path / "r.csv", rating_rows(8, adaptive=5, fixed=2, rnd=3))
    result = st.analyze(st.load_responses(tmp_path / "r.csv"), seed=1)
    q = result["questions"]["responsive"]
    assert q["n_complete"] == 8
    assert q["conditions"]["adaptive"]["mean"] == 5 and q["conditions"]["fixed"]["mean"] == 2
    vs_fixed = q["comparisons"]["adaptive_vs_fixed"]
    assert vs_fixed["mean_diff"] == 3 and vs_fixed["p_exact"] == pytest.approx(2 / 256, abs=1e-4)
    assert vs_fixed["wins"] == 8 and vs_fixed["losses"] == 0
    assert vs_fixed["p_holm"] >= vs_fixed["p_exact"]
    assert q["comparisons"]["adaptive_vs_random"]["mean_diff"] == 2
    assert set(result["questions"]) == set(QS)


def test_analysis_reports_no_effect_honestly():
    rows = rating_rows(8, adaptive=3, fixed=3, rnd=3, jitter=[0, 1, -1, 0, 1, -1, 0, 0])
    q = st.analyze(rows)["questions"]["responsive"]
    comp = q["comparisons"]["adaptive_vs_fixed"]  # identical jitter in every condition
    assert comp["mean_diff"] == 0 and comp["p_exact"] == 1.0
    flat = st.analyze(rating_rows(5, adaptive=3, fixed=3, rnd=3))["questions"]["engaged"]
    assert flat["comparisons"]["adaptive_vs_fixed"]["p_exact"] == 1.0
    assert flat["comparisons"]["adaptive_vs_fixed"]["dz"] == 0.0


def test_analysis_needs_complete_data():
    with pytest.raises(st.StudyError):
        st.analyze([])
    only_two = [r for r in rating_rows(3) if r["condition"] != "random"]
    with pytest.raises(st.StudyError):
        st.analyze(only_two)


def test_markdown_and_chart_output():
    result = st.analyze(rating_rows(6), seed=2)
    md = st.render_markdown(result)
    assert "(primary)" in md and "adaptive vs fixed" in md and "Holm" in md
    assert md.count("| responsive |") == 2  # one comparison row per baseline
    svg = st.chart_for(result)
    assert svg.startswith("<svg") and "adaptive" in svg
    assert st.chart_for({"questions": {}}, "responsive") is None
