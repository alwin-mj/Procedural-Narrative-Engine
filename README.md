<div align="center">

<img src="docs/banner.png" alt="Narrative Engine" width="100%"/>

# Narrative Engine

**A real-time story that reads your face through a webcam and adapts as you feel.**
Smile and the world warms up. Look sad and it turns gentle. No controller, just your expression.

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-vision-5C3EE8)
![License](https://img.shields.io/badge/license-MIT-00D2FF)
![Runs locally](https://img.shields.io/badge/runs-100%25%20local-FF9F3C)

</div>

<!-- TODO: record a 30-60 s demo (Windows: ScreenToGif, Mac: QuickTime + ffmpeg, Linux: Peek), save it as docs/demo.gif,
     then delete this comment and uncomment the line below. A visible demo is the single
     most effective thing on a GitHub project page.
<p align="center"><img src="docs/demo.gif" alt="Demo" width="80%"/></p>
-->

---

## What it does

The webcam feeds a perception pipeline that finds your face, estimates your mood, and smooths it over
time. A director layer uses that mood to pick which version of each story scene you see, while the
background colour, particles and character react to match.

- **Webcam is the only input**, nothing to learn.
- **Four mood variants per scene**, so the same story plays differently for different people.
- **Fully local**: no video is stored or sent anywhere.
- **Works out of the box** with a lightweight detector; heavier ML backends are optional.

## Quick start

```bash
git clone https://github.com/alwin-mj/Procedural-Narrative-Engine.git
cd Procedural-Narrative-Engine

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # lightweight: OpenCV + NumPy only
narrative-engine                     # or: python -m narrative_engine
```

Press **S** to start, **SPACE** for the next scene, **Q** to quit.

### Better emotion detection (optional)

The default install uses a Haar-cascade smile cue (smile = happy, otherwise neutral). For a full
emotion model install an optional backend; it is picked up automatically:

```bash
pip install -e ".[fer]"              # FER (pulls in TensorFlow, large)
# or
pip install -e ".[deepface]"
```

If an optional backend fails to install or load, the app logs a warning and falls back to the next one.
The detector currently in use is shown on screen (`detector: fer / deepface / haar`).

## How it works

```
Webcam -> EmotionDetector -> EmotionSmoother -> Director -> GameWindow
          (FER > DeepFace   (rolling average   (story +     (gradient, particles,
           > Haar smile)     over recent        mood         character, HUD, text)
                             frames)            selection)
```

1. OpenCV grabs a mirrored frame.
2. A Haar cascade finds the largest face (once per frame) and crops it with padding.
3. The first available backend scores emotions; raw labels fold into four story moods
   (`happy`, `uplift`, `sad`, `neutral`).
4. `EmotionSmoother` averages the last few frames so the mood does not flicker.
5. Every few seconds, if a face is visible and confidence is high enough, the director advances the story
   and chooses the mood variant (`director/selector.py`).
6. `GameWindow` renders everything.

## Project structure

```
.
├── pyproject.toml
├── requirements.txt / requirements-optional.txt / requirements-dev.txt
├── src/narrative_engine/
│   ├── __main__.py          CLI entry point
│   ├── app.py               main loop and per-frame logic
│   ├── config.py            all tunable settings + CLI flags
│   ├── perception/
│   │   ├── detector.py      face + emotion backends (FER / DeepFace / Haar)
│   │   ├── smoothing.py     temporal smoothing
│   │   └── motion.py        "walking" detection
│   ├── director/
│   │   ├── story.py         story model, JSON loading + validation
│   │   └── selector.py      which mood variant to use on advance
│   ├── render/
│   │   ├── window.py        background, particles, HUD, text
│   │   └── player_animation.py
│   └── data/stories/arin_adventure.json
├── tests/
└── docs/
```

## Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--camera N` | `0` | camera index |
| `--backend` | `auto` | `auto`, `fer`, `deepface`, `haar` |
| `--story PATH` | bundled | path to a story JSON file |
| `--sprites DIR` | none | folder of PNG frames to replace the default orb |
| `--interval S` | `3.5` | seconds between automatic scene changes |
| `--min-confidence X` | `0.25` | confidence needed to advance |
| `--motion-threshold X` | `7.0` | walking sensitivity |
| `--no-happy-bias` | off | disable steering neutral/sad moods toward warmer scenes |
| `--width / --height` | `960 / 540` | window size |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Defaults live in `src/narrative_engine/config.py`.

## Writing your own story

Stories are JSON, not code. Copy `src/narrative_engine/data/stories/arin_adventure.json`, edit it, and run:

```bash
narrative-engine --story my_story.json
```

```json
{
  "title": "My Story",
  "scenes": [
    {
      "id": "intro",
      "variants": {
        "neutral": "...",
        "happy": "...",
        "uplift": "...",
        "sad": "..."
      }
    }
  ],
  "ending": "..."
}
```

Every scene needs a non-empty line for all four moods; the loader validates this and reports a clear error.

## Custom character

By default the character is a glowing orb that moves faster when you move. To use your own sprite, put
transparent PNG frames in a folder (loaded alphabetically, resized to 160x160) and run
`narrative-engine --sprites path/to/frames`.

## Development

```bash
pip install -e ".[dev]"
pytest                    # unit tests
ruff check .              # lint
black .                   # format
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot open camera" | Close other apps using the webcam, or pass `--camera 1`. |
| Mood always `NEUTRAL` | Check `detector:` on screen. `haar` or `none` means no ML backend is active; install `.[fer]`. Face a light source and move closer. |
| Slow first frames | ML backends load large models at startup. `--backend haar` skips them. |

## Limitations and privacy

- All processing is **local**. No video or images are stored or transmitted.
- Facial expression is not the same as inner emotion. Models can misread people, lighting, glasses and
  camera angles, and accuracy varies across demographics. Treat this as an interaction mechanic, not a
  measure of how someone feels.
- Designed for one face, one viewer, entertainment use only.

## Roadmap

- [x] Package layout, centralised config, CLI, JSON stories
- [ ] Threaded perception (no frame stalls) and a faster landmark-based detector
- [ ] Continuous valence/arousal + per-user calibration
- [ ] Story graph (branching) and a drama manager
- [ ] Optional local-LLM scene writer with authored-text fallback
- [ ] Benchmarks, classifier evaluation and a small user study
- [ ] CI

## Team

| Name | GitHub |
|---|---|
| Alwin John Shajan | [@AlwinJCOde667](https://github.com/AlwinJCOde667) |
| Alwin M J | [@alwin-mj](https://github.com/alwin-mj) |

B.Tech Computer Science & Engineering, Sahrdaya College of Engineering and Technology.

## License

MIT, see [LICENSE](LICENSE).
