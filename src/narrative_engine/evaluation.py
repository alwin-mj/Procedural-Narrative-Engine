"""Evaluation maths: classifier metrics, flicker, onset latency, small-sample stats, SVG charts.

Everything here is a pure function of the data passed in, so it can be unit-tested without a
camera, and has no dependencies beyond NumPy.
"""

from __future__ import annotations

import math
from collections import Counter
from html import escape
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import MOODS

Labels = Tuple[str, ...]


# ---------------------------------------------------------------------- classifier metrics
def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = MOODS
) -> List[List[int]]:
    """Rows are the true label, columns the predicted label (both in ``labels`` order)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    index = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"unknown label in ({t!r}, {p!r}); expected one of {list(labels)}")
        cm[index[t]][index[p]] += 1
    return cm


def classification_report(
    cm: Sequence[Sequence[int]], labels: Sequence[str] = MOODS
) -> Dict[str, Any]:
    """Accuracy plus per-class precision, recall, F1 and support."""
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(labels)))
    per_class: Dict[str, Dict[str, Any]] = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None if (precision is None or recall is None) else 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    f1s = [c["f1"] for c in per_class.values() if c["f1"] is not None]
    recalls = [c["recall"] for c in per_class.values() if c["recall"] is not None]
    return {
        "n": total,
        "accuracy": correct / total if total else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,  # mean recall
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class,
    }


def majority(values: Sequence[str]) -> str:
    """Most common value (ties broken by first appearance)."""
    if not values:
        raise ValueError("empty sequence")
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def take_level_accuracy(takes: Mapping[str, Tuple[str, Sequence[str]]]) -> Optional[float]:
    """Accuracy when each take (one continuous pose) gets the majority vote of its frames.

    Frames within a take are near-duplicates, so the number of *takes*, not frames, is the
    honest sample size.
    """
    if not takes:
        return None
    hits = sum(1 for truth, preds in takes.values() if preds and majority(preds) == truth)
    return hits / len(takes)


# ---------------------------------------------------------------------- flicker and latency
def count_flips(sequence: Sequence[str]) -> int:
    """How many times consecutive entries differ."""
    return sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)


def flips_per_minute(sequence: Sequence[str], duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    return count_flips(sequence) * 60.0 / duration_s


def onset_latencies(
    times: Sequence[float], truth: Sequence[str], predicted: Sequence[str]
) -> List[Tuple[str, Optional[float]]]:
    """Reaction time to each true change of expression.

    For every point where the ground-truth label changes, find how long until the prediction
    first shows the new label (before the next change). ``None`` means it never did.
    """
    if not (len(times) == len(truth) == len(predicted)):
        raise ValueError("times, truth and predicted must have the same length")
    out: List[Tuple[str, Optional[float]]] = []
    n = len(truth)
    starts = [i for i in range(1, n) if truth[i] != truth[i - 1]]
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        label = truth[i]
        hit = next((j for j in range(i, end) if predicted[j] == label), None)
        out.append((label, None if hit is None else float(times[hit] - times[i])))
    return out


# ---------------------------------------------------------------------- statistics
def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """n, mean, sd, se, median, p95, min and max (rounded)."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {k: None for k in ("mean", "sd", "se", "median", "p95", "min", "max")} | {"n": 0}
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "sd": round(sd, 4),
        "se": round(sd / math.sqrt(a.size), 4),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean (seeded, so it is repeatable)."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        raise ValueError("need at least one value")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_sign_flip_pvalue(
    diffs: Sequence[float], max_exact: int = 16, n_mc: int = 20000, seed: int = 0
) -> float:
    """Two-sided p-value for "the mean paired difference is zero" (sign-flip permutation test).

    Under the null the sign of each pair's difference is arbitrary, so every sign assignment is
    equally likely. For up to ``max_exact`` pairs all 2**n assignments are enumerated (an exact
    test); beyond that a seeded Monte-Carlo estimate is used. Suitable for the small samples and
    ordinal ratings of a short user study; it assumes pairs are independent.
    """
    d = np.asarray(list(diffs), dtype=float)
    if d.size == 0:
        raise ValueError("need at least one pair")
    observed = abs(float(d.sum()))
    if not np.any(d):
        return 1.0
    tol = 1e-9 * max(1.0, float(np.abs(d).sum()))
    if d.size <= max_exact:
        bits = (np.arange(2**d.size)[:, None] >> np.arange(d.size)) & 1
        sums = (bits * 2 - 1) @ d
        return float(np.mean(np.abs(sums) >= observed - tol))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_mc, d.size)) * 2 - 1
    hits = int(np.sum(np.abs(signs @ d) >= observed - tol))
    return (hits + 1) / (n_mc + 1)  # add-one: a permutation p-value is never exactly 0


def cohens_dz(diffs: Sequence[float]) -> Optional[float]:
    """Standardised paired effect size (mean difference / SD of differences)."""
    d = np.asarray(list(diffs), dtype=float)
    if d.size < 2:
        return None
    sd = float(d.std(ddof=1))
    if sd == 0:
        return 0.0 if float(d.mean()) == 0 else None  # constant non-zero difference: undefined
    return float(d.mean() / sd)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (controls the chance of any false positive)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------- SVG charts
_FONT = "font-family='Helvetica,Arial,sans-serif'"


def confusion_svg(
    cm: Sequence[Sequence[int]], labels: Sequence[str] = MOODS, title: str = "Confusion matrix"
) -> str:
    """Heat-map of a confusion matrix (colour = share of each true class, text = counts)."""
    n = len(labels)
    cell, left, top = 84, 110, 70
    width, height = left + n * cell + 20, top + n * cell + 60
    out = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' {_FONT} font-size='13'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='16' "
        f"font-weight='bold'>{escape(title)}</text>",
        f"<text x='{left + n * cell / 2}' y='52' text-anchor='middle' fill='#444'>predicted</text>",
        f"<text x='20' y='{top + n * cell / 2}' text-anchor='middle' fill='#444' "
        f"transform='rotate(-90 20 {top + n * cell / 2})'>true</text>",
    ]
    for j, lab in enumerate(labels):
        out.append(
            f"<text x='{left + j * cell + cell / 2}' y='{top - 8}' text-anchor='middle'>"
            f"{escape(lab)}</text>"
        )
    for i, lab in enumerate(labels):
        out.append(
            f"<text x='{left - 8}' y='{top + i * cell + cell / 2 + 4}' text-anchor='end'>"
            f"{escape(lab)}</text>"
        )
        row_total = sum(cm[i]) or 1
        for j in range(n):
            share = cm[i][j] / row_total
            fill = f"rgb({int(255 - share * 215)},{int(255 - share * 150)},255)"
            ink = "white" if share > 0.55 else "#222"
            out.append(
                f"<rect x='{left + j * cell}' y='{top + i * cell}' width='{cell}' "
                f"height='{cell}' fill='{fill}' stroke='white'/>"
                f"<text x='{left + j * cell + cell / 2}' y='{top + i * cell + cell / 2 + 5}' "
                f"text-anchor='middle' fill='{ink}' font-size='16'>{cm[i][j]}</text>"
            )
    out.append("</svg>")
    return "\n".join(out)


def bar_svg(
    title: str,
    groups: Sequence[Mapping[str, Any]],
    ymin: float = 1.0,
    ymax: float = 5.0,
    ylabel: str = "mean rating",
) -> str:
    """Bar chart with 95% CI whiskers. Each group: ``{"label", "mean", "lo", "hi"}``."""
    colours = ["#8a96a3", "#c9a227", "#2d7dd2", "#3a9d5d"]
    w_bar, gap, left, top, plot_h = 90, 40, 70, 60, 260
    width = left + len(groups) * (w_bar + gap) + 20
    height = top + plot_h + 70

    def y_of(v: float) -> float:
        v = min(ymax, max(ymin, v))
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    out = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' {_FONT} font-size='13'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='16' "
        f"font-weight='bold'>{escape(title)}</text>",
        f"<text x='18' y='{top + plot_h / 2}' text-anchor='middle' fill='#444' "
        f"transform='rotate(-90 18 {top + plot_h / 2})'>{escape(ylabel)}</text>",
    ]
    step = 1.0 if ymax - ymin <= 6 else (ymax - ymin) / 5
    tick = ymin
    while tick <= ymax + 1e-9:
        y = y_of(tick)
        out.append(
            f"<line x1='{left}' y1='{y}' x2='{width - 10}' y2='{y}' stroke='#e3e3e3'/>"
            f"<text x='{left - 8}' y='{y + 4}' text-anchor='end' fill='#555'>{tick:g}</text>"
        )
        tick += step
    for k, g in enumerate(groups):
        x = left + gap / 2 + k * (w_bar + gap)
        y_top, y_base = y_of(float(g["mean"])), y_of(ymin)
        cx = x + w_bar / 2
        out.append(
            f"<rect x='{x}' y='{y_top}' width='{w_bar}' height='{y_base - y_top}' "
            f"fill='{colours[k % len(colours)]}'/>"
            f"<line x1='{cx}' y1='{y_of(float(g['lo']))}' x2='{cx}' y2='{y_of(float(g['hi']))}' "
            f"stroke='#222' stroke-width='2'/>"
            f"<line x1='{cx - 8}' y1='{y_of(float(g['lo']))}' x2='{cx + 8}' "
            f"y2='{y_of(float(g['lo']))}' stroke='#222' stroke-width='2'/>"
            f"<line x1='{cx - 8}' y1='{y_of(float(g['hi']))}' x2='{cx + 8}' "
            f"y2='{y_of(float(g['hi']))}' stroke='#222' stroke-width='2'/>"
            f"<text x='{cx}' y='{y_base + 20}' text-anchor='middle'>"
            f"{escape(str(g['label']))}</text>"
            f"<text x='{cx}' y='{y_top - 8 if y_top > top + 14 else y_top + 16}' "
            f"text-anchor='middle' font-weight='bold' fill='"
            f"{'#222' if y_top > top + 14 else 'white'}'>{float(g['mean']):.2f}</text>"
        )
    out.append("</svg>")
    return "\n".join(out)
