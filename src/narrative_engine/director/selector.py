"""Which mood variant to use when the story advances."""

from __future__ import annotations

from typing import Mapping, Optional

WARM_BIAS_MOODS = ("neutral", "sad")


def choose_advance_emotion(
    scores: Mapping[str, float],
    last_chosen: Optional[str],
    last_scores: Optional[Mapping[str, float]],
    mood_bias: float = 0.0,
) -> str:
    """Pick the mood used for the next scene.

    Pure function (no camera, no state) so it can be unit-tested.

    * ``mood_bias`` (default 0 = off) adds that much score to "happy" (or "uplift") when the
      reading is neutral/sad. It deliberately steers the story toward warmer variants, so it is
      off by default; 0.25 reproduces the original prototype's behaviour.
    * A different top mood always wins. If the top mood repeats, any mood that grew since the
      last progression wins; otherwise a close runner-up is allowed to take over for variety.
    """
    if not scores:
        return last_chosen or "neutral"

    eff = dict(scores)
    raw_top = max(scores.items(), key=lambda kv: kv[1])[0]

    if mood_bias > 0 and (raw_top in WARM_BIAS_MOODS or last_chosen in WARM_BIAS_MOODS):
        if "happy" in eff:
            eff["happy"] += mood_bias
        elif "uplift" in eff:
            eff["uplift"] += mood_bias

    ranked = sorted(eff.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second_name, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

    if not last_chosen or top != last_chosen:
        return top

    if last_scores:
        base = last_scores.get(last_chosen, 0.0)
        grew = [(e, s - base) for e, s in eff.items() if e != last_chosen and s - base > 0]
        if grew:
            grew.sort(key=lambda t: t[1], reverse=True)
            return grew[0][0]

    if second_name and (second_score >= top_score * 0.6 or second_score >= 0.45):
        return second_name
    return top
