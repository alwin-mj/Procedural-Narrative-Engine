# Changelog

Versions follow the project's development phases. Dates are not recorded.

## 0.9.0
- Story packs: **The Last Train** and a minimal **Starter** template alongside Arin's Day.
- `--story NAME` selects a bundled story; `narrative-story list`, `check --all --strict`.
- Story linter: flags that are checked but never set, state that is never used, unknown `visited` scenes,
  scenes with identical mood text. Fixed the bundled story's unused `smith` relationship.
- `docs/architecture.md`, `CONTRIBUTING.md`, this changelog, and a tag-triggered release workflow.

## 0.8.0
- Privacy: a notice and remembered consent before the camera opens (`--yes`, `--forget-consent`).
- Always-visible camera indicator; **X** switches the camera off for real (the device is released).
- `--record` saves the window; `narrative-eval gif` makes a README GIF.
- `docs/ethics.md`.

## 0.7.x
- Evaluation tooling: JSONL telemetry with replay, `narrative-eval` (benchmark, classifier accuracy with
  confidence intervals, flicker and reaction delay, LLM timing, counterfactual replay), user-study plan and
  analysis, `docs/evaluation.md` and `docs/user-study-protocol.md`.
- Fixes: LLM output check no longer requires the hero's name when the scene does not use it; rejected answers
  no longer pause the narrator; early "file not found" errors; Python 3.9 dropped (end of life).

## 0.6.0
- Test suite with a coverage floor, GitHub Actions CI (lint, Python matrix), pre-commit, Dependabot.

## 0.5.0
- Threaded capture, perception and rendering; frame dropping and downscaling; graceful degradation; headless
  and video replay with a JSON report.

## 0.4.0
- Branching story graph with state and trajectory endings, drama manager, optional local-LLM narrator.

## 0.3.0
- MediaPipe blendshape detector, continuous valence/arousal, per-user calibration, hysteresis, face-based
  motion.

## 0.2.0
- Restructured into a package with central configuration, a command-line interface and JSON stories.

## 0.1.0
- The original prototype: a webcam emotion detector choosing between pre-written lines.
