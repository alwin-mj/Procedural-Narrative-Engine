import cv2
import numpy as np
from collections import deque

# Primary: FER; Fallback: DeepFace; final fallback: Haar cascade
try:
    from fer import FER
    HAS_FER = True
except ImportError:
    HAS_FER = False

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

# Map raw emotions to simplified categories
def map_emotion(raw):
    mapping = {
        "happy": "happy",
        "surprise": "uplift",
        "neutral": "neutral",
        "sad": "sad",
        "angry": "sad",
        "fear": "sad",
        "disgust": "sad",
        "calm": "neutral",
        "content": "happy",
        "excited": "uplift",
        "bored": "neutral",
        "confused": "neutral",
        "embarrassed": "sad",
        "shocked": "uplift",
        "serious": "neutral",
        "tired": "neutral",
        "worried": "sad",
        "joy": "happy",
        "happiness": "happy",
        "surprised": "uplift",
        "anger": "sad",
        "fearful": "sad",
        "disgusted": "sad",
        "sadness": "sad"
    }
    return mapping.get(str(raw).lower(), "neutral")

class EmotionDetector:
    def __init__(self):
        # buffer stores recent per-frame simplified-distribution dicts
        self.buffer = deque(maxlen=10)  # each entry: { "happy":.., "uplift":.., "sad":.., "neutral":.. }
        self.last = "neutral"
        self.last_conf = 0.0
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # add smile cascade for a lightweight happy cue
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        if HAS_FER:
            # FER can be slow; mtcnn gives better boxes if available
            try:
                self.fer = FER(mtcnn=True)
            except Exception:
                self.fer = FER(mtcnn=False)
        else:
            self.fer = None

    # detect returns: dominant_emotion (simplified), confidence (0..1), box (or None), frame_scores (dict of simplified categories -> 0..1)
    def detect(self, frame):
        # defensive checks
        if frame is None:
            return "neutral", 0.0, None, {"happy":0.0,"uplift":0.0,"sad":0.0,"neutral":1.0}

        # try to detect face first so we can crop and give models only the face
        face_box = None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces):
                # choose largest face
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x, y, w, h = faces[0]
                # expand box slightly (clamp to image)
                pad_w = int(w * 0.12)
                pad_h = int(h * 0.12)
                x0 = max(0, x - pad_w)
                y0 = max(0, y - pad_h)
                x1 = min(frame.shape[1], x + w + pad_w)
                y1 = min(frame.shape[0], y + h + pad_h)
                face_box = (x0, y0, x1 - x0, y1 - y0)
                crop = frame[y0:y1, x0:x1]
            else:
                crop = frame.copy()
        except Exception:
            crop = frame.copy()

        # prepare color-converted images for models
        try:
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb_full = frame
            rgb_crop = crop

        # default empty distribution
        frame_scores = {"happy":0.0,"uplift":0.0,"sad":0.0,"neutral":0.0}

        # Try FER on the face crop if available
        if self.fer:
            try:
                res = self.fer.detect_emotions(rgb_crop)
                if res:
                    em = res[0].get("emotions", {})
                    if em:
                        # FER gives per-emotion scores (0..1)
                        # map raw keys into simplified categories by summing contributions
                        for raw_k, raw_v in em.items():
                            simp = map_emotion(raw_k)
                            frame_scores[simp] = frame_scores.get(simp, 0.0) + float(raw_v)
                        # normalize so the scores sum to 1 (if they don't, scale)
                        total = sum(frame_scores.values())
                        if total > 0:
                            for k in frame_scores:
                                frame_scores[k] = frame_scores[k] / total
                        dom_raw = max(em, key=em.get)
                        conf = float(em.get(dom_raw, 0.0))
                        dom = map_emotion(dom_raw)
                        box = None
                        if face_box:
                            bx, by, bw, bh = face_box
                            # FER may also return box on the crop; try to use it
                            box0 = res[0].get("box", None)
                            if box0:
                                bx0, by0, bw0, bh0 = box0
                                box = (bx + bx0, by + by0, bw0, bh0)
                            else:
                                box = (bx, by, bw, bh)
                        # smile heuristic: if neutral dominant but smile detected in crop, nudge happy
                        if dom == "neutral" and conf < 0.70:
                            try:
                                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                faces_in_crop = self.face_cascade.detectMultiScale(gray_crop, 1.1, 4)
                                for (cx,cy,cw,ch) in faces_in_crop:
                                    mouth_region = gray_crop[cy+int(ch*0.55):cy+ch, cx:cx+cw]
                                    smiles = self.smile_cascade.detectMultiScale(mouth_region, 1.7, 20)
                                    if len(smiles) >= 1:
                                        frame_scores["happy"] = max(frame_scores["happy"], 0.55)
                                        dom = "happy"
                                        conf = max(conf, 0.55)
                                        break
                            except Exception:
                                pass
                        return dom, conf, box, frame_scores
            except Exception:
                pass

        # Try DeepFace on the face crop if available
        if HAS_DEEPFACE:
            try:
                analysis = DeepFace.analyze(rgb_crop, actions=["emotion"], enforce_detection=False, detector_backend='opencv')
                emo_scores = analysis.get("emotion", {})
                if emo_scores:
                    # DeepFace emotion values are often 0..100
                    for raw_k, raw_v in emo_scores.items():
                        simp = map_emotion(raw_k)
                        frame_scores[simp] = frame_scores.get(simp, 0.0) + float(raw_v) / 100.0
                    # normalize
                    total = sum(frame_scores.values())
                    if total > 0:
                        for k in frame_scores:
                            frame_scores[k] = frame_scores[k] / total
                    dom_raw = analysis.get("dominant_emotion", "neutral")
                    conf = float(emo_scores.get(dom_raw, 0.0)) / 100.0
                    dom = map_emotion(dom_raw)
                    box = None
                    if face_box:
                        bx, by, bw, bh = face_box
                        box = (bx, by, bw, bh)
                    # smile heuristic if model unsure
                    if dom == "neutral" and conf < 0.65:
                        try:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            faces_in_crop = self.face_cascade.detectMultiScale(gray_crop, 1.1, 4)
                            for (cx,cy,cw,ch) in faces_in_crop:
                                mouth_region = gray_crop[cy+int(ch*0.55):cy+ch, cx:cx+cw]
                                smiles = self.smile_cascade.detectMultiScale(mouth_region, 1.7, 20)
                                if len(smiles) >= 1:
                                    frame_scores["happy"] = max(frame_scores["happy"], 0.55)
                                    dom = "happy"
                                    conf = max(conf, 0.55)
                                    break
                        except Exception:
                            pass
                    return dom, conf, box, frame_scores
            except Exception:
                pass

        # Fallback: Haar face with smile heuristic (conservative)
        try:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_full = self.face_cascade.detectMultiScale(gray_full, 1.1, 4)
            if len(faces_full):
                (x,y,w,h) = faces_full[0]
                mouth_region = gray_full[y+int(h*0.55):y+h, x:x+w]
                smiles = []
                try:
                    smiles = self.smile_cascade.detectMultiScale(mouth_region, 1.7, 20)
                except Exception:
                    smiles = []
                if len(smiles) >= 1:
                    frame_scores = {"happy":0.8,"uplift":0.0,"sad":0.0,"neutral":0.2}
                    return "happy", 0.55, faces_full[0], frame_scores
                # no clear smile: return last known emotion with light confidence and map to distribution
                fallback_dist = {"happy":0.0,"uplift":0.0,"sad":0.0,"neutral":1.0}
                return self.last, 0.45, faces_full[0], fallback_dist
        except Exception:
            pass

        # final fallback
        return "neutral", 0.0, None, {"happy":0.0,"uplift":0.0,"sad":0.0,"neutral":1.0}

    # detect_stable now aggregates per-frame distributions and returns averaged scores dict
    def detect_stable(self, frame):
        dom, conf, box, frame_scores = self.detect(frame)

        # normalize confidence to 0..1
        try:
            conf = float(conf)
            if conf > 1.0: conf = 1.0
            if conf < 0.0: conf = 0.0
        except Exception:
            conf = 0.0

        # store per-frame simplified distribution in buffer
        self.buffer.append(frame_scores)

        # Take up to last 6 frames to compute averaged distribution
        window = list(self.buffer)[-6:]
        window_size = len(window)
        avg_conf = {"happy":0.0,"uplift":0.0,"sad":0.0,"neutral":0.0}
        counts = {"happy":0,"uplift":0,"sad":0,"neutral":0}
        if window_size > 0:
            for d in window:
                for k in avg_conf:
                    v = float(d.get(k, 0.0))
                    avg_conf[k] += v
                    if v > 0.01:
                        counts[k] += 1
            for k in avg_conf:
                avg_conf[k] = avg_conf[k] / window_size

        # scores returned to caller are the averaged confidences (0..1)
        scores = dict(avg_conf)

        # Decide stable emotion using rules similar to prior heuristic
        stable = max(scores, key=scores.get)
        stable_avg = scores.get(stable, 0.0)
        neutral_score = scores.get("neutral", 0.0)
        stable_count = counts.get(stable, 0)

        # Prefer non-neutral if sustained enough or close to neutral
        if stable != "neutral":
            if not ((stable_count >= 2 and stable_avg >= 0.40) or (scores.get(stable,0.0) >= neutral_score * 0.7)):
                if "neutral" in scores and neutral_score >= 0.2:
                    stable = "neutral"
                    stable_avg = neutral_score
                else:
                    stable = dom
                    stable_avg = conf
        else:
            stable_avg = scores.get("neutral", conf)

        self.last = stable
        self.last_conf = stable_avg
        # return stable, confidence, box, and the averaged scores dictionary
        return stable, stable_avg, box, scores
