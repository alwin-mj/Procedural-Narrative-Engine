# Contributing

## Set up

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # add ,mediapipe for the real face detector
pre-commit install                   # optional: format and lint on every commit
```

## Before you open a pull request

```bash
black .
ruff check --fix .
pytest --cov=narrative_engine --cov-report=term-missing
narrative-story check --all --strict
```

CI runs the same checks on Python 3.10 to 3.12. New behaviour comes with a test; the suite needs no camera,
model or LLM, so please keep it that way (inject fakes instead).

## Add a story

Copy `src/narrative_engine/data/stories/starter.json`, edit it, and run
`narrative-story check yourfile.json --strict`. A story with warnings is usually a mistake (a flag that is
never set, a scene with the same text for every mood). See the README section "Writing your own story".

## Ground rules

- **Privacy first.** Never commit face images, recordings, telemetry logs, or participant data (the
  `.gitignore` already excludes `eval_data/`, `study/`, `logs/` and `*.jsonl`). Do not add anything that sends
  camera data off the machine.
- Do not use this project to assess people at work or school or anyone who cannot say no; see
  `docs/ethics.md`.
- Do not quote accuracy or study numbers you did not measure with the project's own tools.
