# Ethics, privacy and responsible use

This project reads a person's facial expression through a webcam and adapts a story to it. That is
useful for learning and for a fun interaction, and it is also exactly the kind of technology that
deserves care. This page says what the software does with your data, what it deliberately does not
do, how it can be wrong, and where it should not be used.

> This is an engineering document, not legal advice. Laws differ by country and change over time.
> If you plan to use the software with other people (a study, a class, a product), check the rules
> that apply to you and get any approval your institution requires.

## What data the program touches

| Data | Where it goes | Stored? |
|---|---|---|
| Webcam frames | Processed in memory on your computer by OpenCV and MediaPipe | **No.** Discarded after each frame |
| Mood, valence/arousal, scene choices | Used live by the story engine | Only if you pass `--telemetry` (labels and numbers, **never images**) |
| Your face pictures | Saved by `narrative-eval collect` into `eval_data/` for the accuracy evaluation | Yes, locally, only when you run that command. The folder is in `.gitignore` |
| The window, including your webcam preview | Saved by `--record FILE` for demos | Yes, to the file you name, only when you ask |
| A face-model file (about 4 MB) | Downloaded once from Google's public MediaPipe model storage | Cached in `~/.cache/narrative-engine`. `--mp-model` avoids the download |
| Story text for the optional narrator | Sent to **your own** Ollama server (`--llm`, default `localhost`) | Nothing leaves your machine unless you point `--llm-host` elsewhere |

Nothing is uploaded, analysed remotely, or shared. There is no account, no analytics and no
advertising.

## Consent and control

- **A privacy notice is shown before the camera opens**, and the camera stays off unless you agree.
  Saying yes is remembered on that computer until the notice changes. `--yes` accepts for one run
  only (useful for scripts and a researcher's own sessions), and `--forget-consent` deletes the
  remembered choice. Runs that use a video file or a telemetry replay open no camera and ask nothing.
- **A red "CAMERA ON" indicator is always on screen**, including in the blind study mode.
- **Press `X` to switch the camera off.** This releases the device (your camera's light goes out)
  rather than hiding the picture; the story pauses; press `X` again to resume.
- Quitting (`Q`, closing the window, Ctrl+C or a crash) releases the camera.

## Limits of what it can know

- **A facial expression is not a feeling.** People smile when uncomfortable and look neutral when
  delighted; expressions differ across cultures and individuals. The software estimates how a face
  *looks*, never what a person feels.
- **The mapping is hand-set.** The valence/arousal weights are heuristics, not a model trained on
  representative data. See [evaluation.md](evaluation.md) for how to measure them, and note that any
  measurement you make on posed faces of one person says little about other people.
- **It can be wrong more often for some people.** Face and expression models are known to perform
  unevenly across skin tones, lighting, camera quality, glasses, facial hair, head coverings, age,
  and facial differences or paralysis. Calibration helps a person's own resting face but does not fix
  this. Do not read a low reading as evidence about someone.
- **It reads one face at a time**, and it is built for entertainment.

## Where it should not be used

Do not use this software, or code derived from it, to:

- assess, rank or monitor employees, job candidates, students or test-takers;
- infer emotions of people who have not clearly agreed, or who cannot freely say no;
- make decisions that affect someone's opportunities, safety, pay, grades or liberty;
- run covertly (the notice and the camera indicator exist so that it can't).

For context: as I understand it, the EU AI Act (Article 5(1)(f)) prohibits AI systems that infer
people's emotions in the workplace and in educational institutions, with narrow medical and safety
exceptions, and has applied since February 2025. Outside those settings, emotion recognition is
treated as a regulated, high-risk use. Other places have their own privacy and data-protection laws
that treat a face image as personal data. Check what applies to you.

## If you run a user study

- Get consent from every participant **before** they sit down, using the wording in
  [user-study-protocol.md](user-study-protocol.md), and let them withdraw at any time.
- Check whether your institution requires ethics or supervisor approval for research with people.
  A study inside a college that involves students is exactly the situation to ask about first.
- Use anonymous participant codes. Never write names in any file. Keep `logs/` and `study/` out of
  version control (they are already ignored).
- Tell participants afterwards what the conditions were (the protocol includes a debrief).

## Honest reporting

- Publish limitations next to results, including null and negative results.
- Report uncertainty (the tools print confidence intervals) and the sample size.
- Do not describe the software as detecting "emotions" or "mood" without saying it reads facial
  expression and has been tested only as described in the evaluation.

## Possible improvements

- An engagement-only mode that uses head pose and blink rate and never labels an expression.
- Per-person data deletion for study logs, and automatic expiry of old telemetry.
- Testing across more people and conditions, with their informed consent.
