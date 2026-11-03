"""Continuous affect model: facial features -> valence / arousal -> story-mood scores.

* **Valence**: unhappy (-1) .. happy (+1).
* **Arousal**: calm (0) .. excited (+1). Facial cues only reliably show *raised* arousal
  (wide eyes, raised brows, open jaw), so blendshape-based arousal stays in [0, 1].

The weights below are **heuristics, not a trained model**. They are deliberately in one
dataclass so they can be tuned and evaluated (see the roadmap's evaluation phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import MOODS

Scores = Dict[str, float]

FEATURES: Tuple[str, ...] = (
    "smile",
    "cheek",
    "frown",
    "brow_down",
    "brow_up",
    "eye_wide",
    "jaw_open",
    "sneer",
    "press",
)

# Russell circumplex placement of the classic 7 emotion labels: (valence, arousal).
# Used so the FER / DeepFace backends also report continuous valence/arousal.
CIRCUMPLEX: Dict[str, Tuple[float, float]] = {
    "happy": (0.8, 0.4),
    "surprise": (0.4, 0.8),
    "neutral": (0.0, 0.0),
    "sad": (-0.7, -0.4),
    "angry": (-0.6, 0.7),
    "fear": (-0.7, 0.7),
    "disgust": (-0.6, 0.3),
}
_ALIASES = {
    "happiness": "happy",
    "joy": "happy",
    "surprised": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "fearful": "fear",
    "disgusted": "disgust",
}


@dataclass(frozen=True)
class AffectWeights:
    """Linear weights from facial features (each in [0, 1]) to valence and arousal."""

    smile: float = 1.5
    cheek: float = 0.5
    frown: float = -1.3
    sneer: float = -0.9
    brow_down: float = -0.4
    press: float = -0.3
    # arousal
    jaw_open: float = 1.0
    eye_wide: float = 1.2
    brow_up: float = 0.9


DEFAULT_WEIGHTS = AffectWeights()


def _mean(bs: Mapping[str, float], *names: str) -> float:
    return sum(float(bs.get(n, 0.0)) for n in names) / len(names)


def extract_features(blendshapes: Mapping[str, float]) -> Dict[str, float]:
    """Collapse MediaPipe's 52 blendshape scores into nine interpretable features in [0, 1]."""
    return {
        "smile": _mean(blendshapes, "mouthSmileLeft", "mouthSmileRight"),
        "cheek": _mean(blendshapes, "cheekSquintLeft", "cheekSquintRight"),
        "frown": _mean(blendshapes, "mouthFrownLeft", "mouthFrownRight"),
        "brow_down": _mean(blendshapes, "browDownLeft", "browDownRight"),
        "brow_up": _mean(blendshapes, "browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
        "eye_wide": _mean(blendshapes, "eyeWideLeft", "eyeWideRight"),
        "jaw_open": float(blendshapes.get("jawOpen", 0.0)),
        "sneer": _mean(blendshapes, "noseSneerLeft", "noseSneerRight"),
        "press": _mean(blendshapes, "mouthPressLeft", "mouthPressRight"),
    }


def baseline_from_samples(samples: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Per-user neutral baseline: the median of each feature over calibration samples."""
    if not samples:
        raise ValueError("need at least one sample")
    return {k: float(np.median([s.get(k, 0.0) for s in samples])) for k in FEATURES}


def remove_baseline(
    features: Mapping[str, float], baseline: Optional[Mapping[str, float]]
) -> Dict[str, float]:
    """Subtract the resting face and rescale so a full expression still reaches 1.0."""
    if not baseline:
        return dict(features)
    out: Dict[str, float] = {}
    for key, value in features.items():
        b = min(0.95, max(0.0, float(baseline.get(key, 0.0))))
        out[key] = max(0.0, float(value) - b) / (1.0 - b)
    return out


def estimate_affect(
    features: Mapping[str, float],
    baseline: Optional[Mapping[str, float]] = None,
    weights: AffectWeights = DEFAULT_WEIGHTS,
) -> Tuple[float, float]:
    """Return ``(valence, arousal)`` with valence in [-1, 1] and arousal in [0, 1]."""
    f = remove_baseline(features, baseline)
    g = lambda k: float(f.get(k, 0.0))  # noqa: E731
    valence = (
        weights.smile * g("smile")
        + weights.cheek * g("cheek")
        + weights.frown * g("frown")
        + weights.sneer * g("sneer")
        + weights.brow_down * g("brow_down")
        + weights.press * g("press")
    )
    arousal = (
        weights.jaw_open * g("jaw_open")
        + weights.eye_wide * g("eye_wide")
        + weights.brow_up * g("brow_up")
    )
    return float(np.clip(valence, -1.0, 1.0)), float(np.clip(arousal, 0.0, 1.0))


def scores_from_affect(valence: float, arousal: float) -> Scores:
    """Turn (valence, arousal) into a soft distribution over the four story moods.

    happy  : positive valence            sad    : negative valence
    uplift : raised arousal (surprise / excitement), damped when valence is clearly negative
    neutral: whatever is left when no mood is strongly expressed
    """
    happy = max(0.0, valence)
    sad = max(0.0, -valence)
    uplift = max(0.0, arousal) * (1.0 if valence >= -0.2 else 0.3)
    peak = max(happy, sad, uplift)
    neutral = max(0.0, 1.0 - 1.5 * peak) + 0.05
    raw = {"happy": happy, "uplift": uplift, "sad": sad, "neutral": neutral}
    total = sum(raw.values())
    return {m: raw[m] / total for m in MOODS}


def circumplex_va(probs: Mapping[str, float]) -> Tuple[float, float]:
    """Expected (valence, arousal) of a classic 7-label emotion distribution."""
    acc_v = acc_a = total = 0.0
    for label, p in probs.items():
        key = _ALIASES.get(str(label).lower(), str(label).lower())
        if key in CIRCUMPLEX and p > 0:
            v, a = CIRCUMPLEX[key]
            acc_v += float(p) * v
            acc_a += float(p) * a
            total += float(p)
    if total <= 0:
        return 0.0, 0.0
    return acc_v / total, acc_a / total
