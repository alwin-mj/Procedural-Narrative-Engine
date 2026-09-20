<div align="center">

<img src="https://capsule-render.vercel.app/api?type=slice&color=0:FF4EC7,50:8C5AFF,100:00D2FF&height=260&section=header&text=NARRATIVE%20ENGINE&fontSize=64&fontColor=ffffff&fontAlignY=40&desc=The%20story%20that%20watches%20you%20back&descSize=22&descAlignY=62&animation=twinkling" width="100%" alt="Narrative Engine"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=1000&color=FF4EC7&center=true&vCenter=true&width=700&height=50&lines=Smile+%E2%86%92+the+world+warms+up;Look+sad+%E2%86%92+the+story+turns+gentle;Look+surprised+%E2%86%92+a+mystery+begins;No+controller.+Just+your+face." alt="Typing animation of how the story reacts to your face"/>

<br/><br/>

**A real-time story that reads your face through a webcam and rewrites itself as you feel. Smile and the world warms up. Look sad and it turns gentle. No controller, no dialogue choices, just your expression.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![FER](https://img.shields.io/badge/Emotion-FER-FF4EC7?style=for-the-badge)](https://github.com/justinshenk/fer)
[![DeepFace](https://img.shields.io/badge/Fallback-DeepFace-8C5AFF?style=for-the-badge)](https://github.com/serengil/deepface)
[![Local](https://img.shields.io/badge/Runs-100%25%20Local-FF9F3C?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-00D2FF?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

> *"What if the story could see how you feel?"*

<br/>

<!-- Add a screenshot or GIF here once you have one:
![Demo](docs/demo.gif)
-->

</div>

---

## 💡 The Idea in One Sentence

The AI Director watches your face, classifies your mood in real time, and uses it to choose which version of each story scene you see, while the background, particles, and mood of the world shift to match.

No buttons to press. No menus. Just react.

---

## 🎯 Why This Exists

Most interactive stories ask you to *tell* them what you want through menus, dialogue trees, or button presses. But the strongest reactions are the ones you never type.

This project explores a different input: **your emotional state**.

- Webcam is the only controller, so there is nothing to learn
- Every scene has four emotional variants, so the same story plays differently for different people
- A director layer decides when to advance and which mood to follow, instead of a fixed script
- Everything runs locally, so no video is stored or sent anywhere

---

## ⚙️ How It Works

```
┌──────────────┐     ┌────────────────────────────────────────────────┐
│    Webcam    │────▶│                EmotionDetector                 │
│ (live frames)│     │  FER  →  DeepFace  →  Haar cascade (fallback)  │
└──────────────┘     └───────────────────────┬────────────────────────┘
                                             │  raw emotions mapped to
                                             │  happy · uplift · sad · neutral
                                             ▼
                     ┌────────────────────────────────────────────────┐
                     │           Temporal smoothing                   │
                     │      rolling average over recent frames        │
                     └───────────────────────┬────────────────────────┘
                                             │  stable emotion + scores
                                             ▼
                     ┌────────────────────────────────────────────────┐
                     │        AI Director  (StorySystem)              │
                     │   picks scene variant · decides when to advance│
                     └───────────────────────┬────────────────────────┘
                                             │
                                             ▼
                     ┌────────────────────────────────────────────────┐
                     │                 GameWindow                     │
                     │  gradient · particles · character · story · HUD│
                     └────────────────────────────────────────────────┘
```

**Detection pipeline:**
1. OpenCV grabs a mirrored webcam frame
2. A Haar cascade finds the largest face and crops it with a little padding
3. FER (or DeepFace, or a smile heuristic) scores the emotions on that crop
4. Raw emotions are folded into four categories
5. Scores are averaged over the last few frames so the result does not flicker
6. A smile detector nudges "neutral" toward "happy" when the model is unsure

**Director pipeline:**
1. Current scene line is chosen from the stable emotion
2. Every few seconds, if a face is present and confidence is high enough, the story advances
3. The director biases gently toward warmer outcomes when you have been neutral or sad
4. If your emotion changes mid-scene, a short reaction line appears without skipping the scene

---

## 📊 Key Specifications

| Parameter | Value |
|-----------|-------|
| Input | Webcam (`VideoCapture(0)`) |
| Emotion categories | 4 (`happy`, `uplift`, `sad`, `neutral`) |
| Story structure | 8 scenes × 4 emotional variants + ending |
| Detector chain | FER → DeepFace → Haar face + smile heuristic |
| Smoothing | Rolling buffer of 10 frames, averaged over the latest 6 |
| Auto-advance interval | 3.5 seconds |
| Min. confidence to advance | 0.25 |
| Window size | 960 × 540 |
| Motion threshold (walking) | Mean frame difference > 7.0 |
| History display | Last 30 detected emotions |
| Character sprite | 160 × 160 px at 10 FPS (optional) |

---

## 🎨 Emotion Mapping

Detectors report many raw emotions. The engine folds them into four story moods.

| Raw emotion | Story mood | Effect on the world |
|-------------|-----------|---------------------|
| happy, joy, content | **happy** | Green gradient, warm and friendly scenes |
| surprise, excited, shocked | **uplift** | Bright blue gradient, particles drift upward, mysterious and hopeful scenes |
| sad, angry, fear, disgust, worried | **sad** | Purple gradient, gentle and comforting scenes |
| neutral, calm, bored, serious | **neutral** | Grey gradient, calm and observational scenes |

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **Live emotion detection** | FER as primary, DeepFace as second choice, Haar smile cue as last resort |
| **Never crashes on missing models** | Each detector is optional; the next one in the chain takes over |
| **Adaptive storyline** | Each scene has a distinct line for every mood |
| **Emotion-reactive visuals** | Background gradient, vignette, and particles change with your mood |
| **Motion-based walking** | Moving in front of the camera speeds up the on-screen character |
| **Live HUD** | Circular webcam preview, emotion + confidence, emotion history bars |
| **Custom character** | Drop transparent PNG frames in a folder to replace the default orb |
| **Fully local** | No network calls at runtime, no video stored |

---

## 📁 Project Structure

```
Procedural-Narrative-Engine/
│
├── simple_ai_director.py   Entry point
│                           Story scenes, director logic, main loop
│
├── emotion_detector.py     EmotionDetector class
│                           FER / DeepFace / Haar chain, smoothing
│
├── game_window.py          GameWindow class
│                           Gradients, particles, HUD, story text box
│
├── player_animation.py     PlayerAnimation class
│                           Sprite frame loader, alpha-blended overlay
│
├── requirements.txt        Python dependencies
└── __init__.py             Package marker
```

---

## 🚀 Installation

### Requirements

- Python 3.8+
- A working webcam
- Even lighting on your face for the best results

```bash
git clone https://github.com/alwin-mj/Procedural-Narrative-Engine.git
cd Procedural-Narrative-Engine

# Recommended: virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `fer` and `deepface` depend on TensorFlow, so the first install is large. Model weights may also download on the first run.

### Run

```bash
python simple_ai_director.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `S` | Start the adventure |
| `SPACE` | Manually advance to the next scene |
| `Q` | Quit |

---

## 🕹️ Playing

1. Sit in front of the webcam with your face clearly visible
2. Press **S** to begin
3. Watch the first scene, and let your face do the talking
4. The story advances by itself every few seconds while you are detected
5. Change your expression and the scene text, colours, and particles respond
6. After the last scene, the story closes with an ending line

---

## 🧍 Custom Character (Optional)

Place transparent PNG frames here to replace the default glowing orb:

```
assets/player/walk/
├── frame_01.png
├── frame_02.png
└── ...
```

Frames are loaded in alphabetical order and resized to 160 × 160.

---

## ✍️ Writing Your Own Story

Edit the `SCENES` list in `simple_ai_director.py`. Every scene needs one line per mood:

```python
{
    "neutral": "...",
    "happy":   "...",
    "uplift":  "...",
    "sad":     "..."
}
```

The engine handles detection, smoothing, progression, and visuals for you.

---

## 🔧 Tuning

| What | Where | Default |
|------|-------|---------|
| Window size | `GameWindow(width, height)` in `main()` | `960 × 540` |
| Auto-advance interval | `now - last > 3.5` in `main()` | `3.5 s` |
| Min. confidence | `conf > 0.25` in `main()` | `0.25` |
| Walking sensitivity | `motion > 7.0` in `main()` | `7.0` |
| Smoothing window | `deque(maxlen=10)` and `[-6:]` in `EmotionDetector` | `6 frames` |

---

## 🐞 Known Issues & Fixes

### "Error: cannot open camera"

**Cause:** The app opens camera index `0`. Another app may be using it, or your webcam has a different index.

**Fix:** Close other apps that use the camera, or change `cv2.VideoCapture(0)` in `simple_ai_director.py` to `1`.

### Emotion always shows NEUTRAL

**Cause:** Poor lighting, face too small, or the ML models failed to load and only the Haar fallback is running.

**Fix:** Face a light source, move closer, and check that `fer` installed correctly (`pip install fer`).

### DeepFace fallback not working

**Cause:** Newer DeepFace versions return a list from `analyze()` instead of a single dict, which the current fallback code does not unpack. The detector then drops to the Haar fallback.

**Fix:** Use FER as the primary detector, or update the DeepFace branch to read `analysis[0]`.

### Slow or laggy first frames

**Cause:** FER and DeepFace load large models on startup.

**Fix:** Wait a few seconds. Set `FER(mtcnn=False)` in `emotion_detector.py` for a faster (but less accurate) face box.

---

## 🔮 Future Roadmap

- **Voice narration** — text-to-speech that matches the mood of each scene
- **Branching paths** — multiple routes and endings, not just one line per scene
- **LLM-generated scenes** — replace pre-written variants with generated ones
- **Audio emotion** — tone of voice as a second signal alongside the face
- **Session recap** — save a "story of your session" at the end
- **Music layer** — adaptive soundtrack that shifts with detected mood

---

## 🔒 Privacy & Limitations

- All processing happens **locally**. No video or images are stored or sent anywhere.
- Facial emotion recognition is approximate and can misread people, lighting, and camera angles. Treat it as an interaction mechanic, not a measure of how someone truly feels.

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Alwin John Shajan | [@AlwinJCOde667](https://github.com/AlwinJCOde667) |
| Alwin M J | [@alwin-mj](https://github.com/alwin-mj) |

B.Tech Computer Science & Engineering, Sahrdaya College of Engineering and Technology

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Narrative Engine** — because the best controller is the one you never have to hold.

*Built with Python · OpenCV · FER · DeepFace · NumPy*

<img src="https://capsule-render.vercel.app/api?type=slice&color=0:00D2FF,50:8C5AFF,100:FF4EC7&height=120&section=footer" width="100%" alt=""/>

</div>
