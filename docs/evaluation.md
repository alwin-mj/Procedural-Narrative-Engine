# Evaluation

> **Status: no results yet.** Every number in this document must come from the commands below,
> run on your own recordings and participants. Cells marked `TBD` are placeholders; replace them
> with measured values, or delete the claim. Nothing here has been measured on a real camera.

All commands are part of the `narrative-eval` tool (`pip install -e .`). Raw faces and participant
sheets stay out of git (`eval_data/` and `study/` are in `.gitignore`); only aggregate results go
in `docs/eval/`.

| Question | Command | Output |
|---|---|---|
| How fast is each stage, per backend? | `narrative-eval benchmark` | `docs/eval/benchmark.md` |
| How accurate is each backend on my face? | `narrative-eval classifier` | `docs/eval/classifier.md`, confusion matrices |
| Does smoothing reduce flicker, and what does it cost? | `narrative-eval video` | flips per minute, reaction latency |
| How fast and reliable is the LLM narrator? | `narrative-eval llm` | time to first token, fallback rate |
| Does the adaptive director feel more responsive? | `narrative-eval study-plan` / `study-analyze` | `docs/eval/study_*.svg` |
| What would other policies have done for the same viewer? | `narrative-eval counterfactual` | route table |

## Setup (fill this in once)

| Item | Value |
|---|---|
| Machine (CPU, RAM, OS) | TBD |
| Webcam and resolution | TBD |
| Python / OpenCV / MediaPipe versions | TBD |
| Project commit hash | TBD |
| Date | TBD |

## 1. System performance

```bash
# record a 30 s clip of yourself first (any phone or webcam recording works), then:
narrative-eval benchmark --video my_clip.mp4 --backends haar,mediapipe,fer,deepface
```

Each backend runs in its own process so memory figures are not mixed together. Backends that are not
installed are skipped. The `--threaded-seconds` option adds camera-to-result latency from the real
threaded pipeline.

| Backend | Face p50 (ms) | Inference p50 / p95 (ms) | Smoothing p50 (ms) | Render p50 (ms) | Detector FPS (1 thread) | CPU (1 core) | Peak RSS (MB) | Install (MB) |
|---|---|---|---|---|---|---|---|---|
| haar | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| mediapipe | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| fer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| deepface | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Camera-to-result latency (threaded pipeline, MediaPipe): p50 `TBD` ms, p95 `TBD` ms. Add up to one
render interval (about 33 ms) to get the time until it reaches the screen.

*Notes: "face" is the Haar face search (MediaPipe finds the face inside its own model, so its face
cost is counted under inference). Install size is an estimate that includes dependencies. Frame
times from a video exclude the camera's own exposure time.*

## 2. Classifier accuracy

### Collecting data

Record **posed** expressions of yourself under different lighting. Frames within one take are almost
identical, so the number of **takes** (not frames) is your real sample size.

```bash
narrative-eval collect --subject me --lighting bright --takes-per-label 3
narrative-eval collect --subject me --lighting dim    --takes-per-label 3
narrative-eval collect --subject me --lighting backlit --takes-per-label 3
```

Each take is 3 seconds at 5 frames per second (15 frames). Aim for at least 3 takes per label per
lighting. For calibration, the first `neutral` take of each lighting group is used, so keep it a
genuinely neutral face with at least 10 frames showing the face.

### Running the evaluation

```bash
narrative-eval classifier --data eval_data --backends haar,mediapipe,fer,deepface --calibrate
```

With `--calibrate`, MediaPipe is evaluated twice on the **same held-out frames**: without and with
per-lighting calibration. The calibration take itself is excluded from both.

| Backend | Frames | Takes | Face found | Accuracy | Balanced accuracy | Macro F1 | Take-level accuracy |
|---|---|---|---|---|---|---|---|
| haar | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| mediapipe | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| mediapipe + calibration | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| fer | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| deepface | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Confusion matrices: generated as `docs/eval/confusion_<backend>.svg`. Embed them here, for example
`![mediapipe](eval/confusion_mediapipe.svg)`.

Observations (write after looking at the matrices, not before): `TBD`.

## 3. Smoothing: flicker and delay

Record 60 to 90 seconds of natural, varied expressions. Write a timeline of what you actually did
(`start_s,end_s,label` per line) by watching the clip back, then:

```bash
narrative-eval video my_clip.mp4 --backend mediapipe --timeline timeline.csv --out docs/eval/video.json
```

| Stream | Mood changes per minute | Accuracy vs timeline | Median reaction delay (s) | Missed changes |
|---|---|---|---|---|
| raw (no smoothing) | TBD | TBD | TBD | TBD |
| smoothing + hysteresis | TBD | TBD | TBD | TBD |

Flicker reduction: `TBD` %. The delay column is the price of that stability, so report both.

## 4. LLM narrator (optional)

```bash
ollama pull llama3.2
narrative-eval llm --model llama3.2 --beats 12
```

| Model | Time to first token p50 / p95 (ms) | Total time p50 / p95 (ms) | Fallback rate |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

Fallback rate counts request errors (timeouts, server down) plus answers rejected by the checks (lists,
chatty preamble, refusal, markdown, or the hero's name missing when the scene names him). The command
also prints the reasons for rejections, which is worth reading before blaming the model.

## 5. User study

Full protocol: [user-study-protocol.md](user-study-protocol.md).

```bash
narrative-eval study-plan --participants 12 --out study
# ... run the sessions, fill study/responses.csv ...
narrative-eval study-analyze study/responses.csv --out docs/eval
```

| Item | fixed | random | adaptive |
|---|---|---|---|
| Responsive (primary), mean [95% CI] | TBD | TBD | TBD |
| Engaged | TBD | TBD | TBD |
| Coherent | TBD | TBD | TBD |
| Would play again | TBD | TBD | TBD |
| Reacted to my expression | TBD | TBD | TBD |

Participants: `TBD` (complete cases `TBD`). Primary comparison, adaptive vs fixed: mean difference
`TBD`, exact p `TBD`, Holm-adjusted `TBD`. Adaptive vs random: `TBD`.

![responsive](eval/study_responsive.svg)

## 6. Threats to validity (keep these in the write-up)

- **One subject, posed expressions.** Accuracy on your own acted faces is not accuracy on other
  people or on spontaneous emotion. Facial expression is not the same as inner feeling.
- **Correlated frames.** Report take-level numbers next to frame-level numbers.
- **Hand-set weights.** The valence/arousal formulas were not trained; any accuracy you measure
  reflects those choices, tuned (if at all) on the same person.
- **Small sample, ordinal ratings.** The study uses an exact paired sign-flip test and bootstrap
  intervals because they make few assumptions, but with 8 to 12 classmates only large effects are
  detectable, and classmates may be friendly or curious about the project.
- **Short, novel experience.** Each run is about a minute; ratings may reflect novelty.
- **Imperfect blinding.** In the fixed condition people may notice the same route repeating.
- **Replay is not interactive.** Counterfactual replays reuse recorded reactions; a real viewer
  would react differently to a different story.
- **Hardware dependence.** Latency and FPS numbers hold for the machine listed above only.

## 7. What you can honestly claim

Only state a result once the matching table above is filled with *your* measured values.

| After you have measured... | A fair resume or README line |
|---|---|
| Benchmark | "Real-time pipeline: p95 inference `X` ms, `Y` FPS render with a threaded detector on `<machine>`." |
| Calibration | "Per-user calibration changed balanced accuracy from `A`% to `B`% on `N` held-out frames (`T` takes, one subject)." |
| Smoothing | "EMA + hysteresis cut mood flicker by `X`% at a median `Y` s reaction delay." |
| LLM | "Local LLM narrator with `F`% fallback rate and `T` ms p95 time to first token." |
| Study | "Within-subject study (n=`N`): adaptive rated more responsive than fixed (mean diff `D`, exact p=`P`)." (Only if the result supports it. Report null results too.) |

Do not claim: emotion recognition "accuracy" without saying it is posed, single-subject data; a
study result from fewer than 6 participants; a significant effect that did not survive the Holm
correction; or any number you did not generate with these commands.
