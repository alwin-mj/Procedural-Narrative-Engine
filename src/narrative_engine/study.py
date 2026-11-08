"""User-study support: counterbalanced run plans, response checks and paired analysis.

Design (within-subject): every participant plays the story three times, once per condition,
in a counterbalanced order, and rates each run on 1-5 Likert items.

* ``fixed``    - the same route and neutral wording for everyone (no adaptation)
* ``random``   - random routes and mood variants (reacts to nothing)
* ``adaptive`` - the real engine

The **primary outcome** is decided *before* looking at the data: the "responsive" item,
adaptive vs fixed and adaptive vs random, with Holm correction across those two tests. All
other items are exploratory.
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .evaluation import (
    bar_svg,
    bootstrap_ci,
    cohens_dz,
    holm_adjust,
    paired_sign_flip_pvalue,
    summarize,
)

CONDITIONS = ("fixed", "random", "adaptive")
QUESTIONS: Dict[str, str] = {
    "responsive": "The story felt responsive to me.",
    "engaged": "I felt engaged while playing.",
    "coherent": "The story felt coherent.",
    "replay": "I would play this again.",
    "reacted": "The story reacted to my expression.",
}
PRIMARY = "responsive"
BASELINES = ("fixed", "random")
LIKERT = (1, 2, 3, 4, 5)
RESPONSE_FIELDS = ["participant", "condition", "run", *QUESTIONS, "comment"]


class StudyError(ValueError):
    """Invalid study input (bad plan parameters or response data)."""


# ---------------------------------------------------------------------- plan
def make_plan(
    n_participants: int, seed: int = 0, interval_s: float = 5.0, telemetry_dir: str = "logs/study"
) -> List[Dict[str, Any]]:
    """Counterbalanced run order plus the exact command for every run.

    With three conditions there are six possible orders; participants are cycled through a
    shuffled copy of them, so every order is used equally often (when n is a multiple of 6) and
    no condition is systematically first. Order effects are therefore balanced, not ignored.
    """
    if n_participants < 1:
        raise StudyError("need at least one participant")
    rng = random.Random(seed)
    orders = list(itertools.permutations(CONDITIONS))
    rng.shuffle(orders)
    rows: List[Dict[str, Any]] = []
    for i in range(n_participants):
        pid = f"P{i + 1:02d}"
        order = orders[i % len(orders)]
        for run, cond in enumerate(order, 1):
            rows.append(
                {
                    "participant": pid,
                    "run": run,
                    "condition": cond,
                    "command": (
                        f"narrative-engine --study --mode {cond} --participant {pid} "
                        f"--interval {interval_s:g} --seed {seed + i * 10 + run} "
                        f"--telemetry {telemetry_dir}"
                    ),
                }
            )
    return rows


def write_plan(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "run", "condition", "command"])
        w.writeheader()
        w.writerows(rows)


def write_response_template(plan: Sequence[Mapping[str, Any]], path: Path) -> None:
    """A blank response sheet with one row per planned run, ready to fill in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESPONSE_FIELDS)
        w.writeheader()
        for row in plan:
            keep = ("participant", "condition", "run")
            w.writerow({k: row[k] for k in keep})


# ---------------------------------------------------------------------- responses
def load_responses(path: Path) -> List[Dict[str, Any]]:
    """Read and validate a response sheet (1-5 ratings, known conditions, no duplicates)."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise StudyError(f"cannot read {path}: {exc}") from exc
    seen = set()
    clean: List[Dict[str, Any]] = []
    for n, row in enumerate(rows, 2):  # line 1 is the header
        pid = (row.get("participant") or "").strip()
        cond = (row.get("condition") or "").strip().lower()
        if not pid:
            raise StudyError(f"line {n}: missing participant")
        if cond not in CONDITIONS:
            raise StudyError(f"line {n}: condition must be one of {CONDITIONS}, got {cond!r}")
        if (pid, cond) in seen:
            raise StudyError(f"line {n}: {pid} has two rows for condition {cond!r}")
        seen.add((pid, cond))
        rec: Dict[str, Any] = {
            "participant": pid,
            "condition": cond,
            "comment": row.get("comment") or "",
        }
        blanks = 0
        for q in QUESTIONS:
            raw = (row.get(q) or "").strip()
            if not raw:
                rec[q] = None
                blanks += 1
                continue
            try:
                val = int(raw)
            except ValueError as exc:
                raise StudyError(f"line {n}: {q} must be a whole number 1-5, got {raw!r}") from exc
            if val not in LIKERT:
                raise StudyError(f"line {n}: {q} must be 1-5, got {val}")
            rec[q] = val
        if blanks == len(QUESTIONS):
            continue  # an unfilled template row: not data yet
        clean.append(rec)
    return clean


def complete_participants(rows: Sequence[Mapping[str, Any]], question: str) -> List[str]:
    """Participants who rated ``question`` in all three conditions (needed for paired tests)."""
    by_pid: Dict[str, set] = {}
    for r in rows:
        if r.get(question) is not None:
            by_pid.setdefault(r["participant"], set()).add(r["condition"])
    return sorted(p for p, conds in by_pid.items() if conds >= set(CONDITIONS))


# ---------------------------------------------------------------------- analysis
def _ratings(
    rows: Sequence[Mapping[str, Any]], question: str, pids: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {c: {} for c in CONDITIONS}
    for r in rows:
        if r["participant"] in pids and r.get(question) is not None:
            table[r["condition"]][r["participant"]] = r[question]
    return table


def analyze(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> Dict[str, Any]:
    """Per-question summaries and paired adaptive-vs-baseline comparisons (complete cases)."""
    if not rows:
        raise StudyError("no completed responses to analyse")
    result: Dict[str, Any] = {"questions": {}, "n_rows": len(rows)}
    for q in QUESTIONS:
        pids = complete_participants(rows, q)
        if not pids:
            continue
        table = _ratings(rows, q, pids)
        entry: Dict[str, Any] = {"n_complete": len(pids), "conditions": {}, "comparisons": {}}
        for c in CONDITIONS:
            vals = [table[c][p] for p in pids]
            lo, hi = bootstrap_ci(vals, seed=seed)
            entry["conditions"][c] = {**summarize(vals), "ci95": [round(lo, 3), round(hi, 3)]}
        raw_p = []
        for base in BASELINES:
            diffs = [table["adaptive"][p] - table[base][p] for p in pids]
            lo, hi = bootstrap_ci(diffs, seed=seed)
            p_val = paired_sign_flip_pvalue(diffs, seed=seed)
            raw_p.append(p_val)
            entry["comparisons"][f"adaptive_vs_{base}"] = {
                "mean_diff": round(sum(diffs) / len(diffs), 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "p_exact": round(p_val, 4),
                "dz": None if cohens_dz(diffs) is None else round(cohens_dz(diffs), 3),
                "wins": sum(1 for d in diffs if d > 0),
                "ties": sum(1 for d in diffs if d == 0),
                "losses": sum(1 for d in diffs if d < 0),
            }
        for comp, p_holm in zip(entry["comparisons"].values(), holm_adjust(raw_p)):
            comp["p_holm"] = round(p_holm, 4)
        result["questions"][q] = entry
    if not result["questions"]:
        raise StudyError("no participant has ratings for all three conditions yet")
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    """Tables ready to paste into docs/evaluation.md."""
    lines = [
        "| Question | " + " | ".join(f"{c} (mean [95% CI])" for c in CONDITIONS) + " | n |",
        "|---|" + "---|" * (len(CONDITIONS) + 1),
    ]
    for q, e in result["questions"].items():
        cells = []
        for c in CONDITIONS:
            s = e["conditions"][c]
            cells.append(f"{s['mean']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
        tag = " (primary)" if q == PRIMARY else ""
        lines.append(f"| {QUESTIONS[q]}{tag} | " + " | ".join(cells) + f" | {e['n_complete']} |")
    lines += [
        "",
        "| Question | Comparison | Mean diff [95% CI] | wins/ties/losses | p (exact) "
        "| p (Holm) | dz |",
        "|---|---|---|---|---|---|---|",
    ]
    for q, e in result["questions"].items():
        for name, c in e["comparisons"].items():
            dz = "n/a" if c["dz"] is None else f"{c['dz']:.2f}"
            lines.append(
                f"| {q} | {name.replace('_', ' ')} | {c['mean_diff']:+.2f} "
                f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | "
                f"{c['wins']}/{c['ties']}/{c['losses']} | {c['p_exact']:.3f} | "
                f"{c['p_holm']:.3f} | {dz} |"
            )
    lines += [
        "",
        f"Primary outcome: **{PRIMARY}** (adaptive vs each baseline, Holm-corrected over those two "
        "tests). Every other row is exploratory and uncorrected across questions.",
    ]
    return "\n".join(lines)


def chart_for(result: Mapping[str, Any], question: str = PRIMARY) -> Optional[str]:
    """SVG bar chart (mean with 95% CI) for one question, or ``None`` if it has no data."""
    entry = result["questions"].get(question)
    if entry is None:
        return None
    groups = [
        {
            "label": c,
            "mean": entry["conditions"][c]["mean"],
            "lo": entry["conditions"][c]["ci95"][0],
            "hi": entry["conditions"][c]["ci95"][1],
        }
        for c in CONDITIONS
    ]
    return bar_svg(f"{QUESTIONS[question]} (n={entry['n_complete']})", groups)
