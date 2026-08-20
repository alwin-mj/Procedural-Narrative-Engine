import cv2
import time
import os
import sys
import numpy as np
from emotion_detector import EmotionDetector
from game_window import GameWindow
from player_animation import PlayerAnimation
from collections import deque

# Replace disconnected tracks with a scene-based adventure about a boy.
# Each scene has variants for neutral/happy/uplift/sad so the narrative adapts to detected emotion.
SCENES = [
    {
        "neutral": "Arin wakes at the edge of town and steps out into the thin morning light, pockets empty but curiosity full.",
        "happy": "Arin wakes with a skip in his step, humming a tune as he ties his boots and greets the day with a grin.",
        "uplift": "Arin wakes to distant music — a festival is being set up. Today might hold an unexpected encounter.",
        "sad": "Arin wakes slowly, the cobbles feel colder. He takes a breath and resolutely steps toward the town."
    },



    {
        "neutral": "He wanders into the market where stalls smell of spice and wood; merchants call softly as they arrange wares.",
        "happy": "He bargains with a laughing spice-seller, trading a joke for a bright orange pouch that smells of sun.",
        "uplift": "A traveling minstrel sings a strange melody; Arin catches a line that feels like a clue to something waiting.",
        "sad": "A stall owner nods kindly but is busy; Arin buys a single apple and watches children dart between crates."
    },
    {
        "neutral": "At the square he meets a woman with ink-stained fingers who sketches maps for a small guild.",
        "happy": "She gifts him a folded map with a flourish; their shared smile feels like the start of a friendship.",
        "uplift": "She points to a hidden lane on the map and whispers of a rooftop garden that appears only at dusk.",
        "sad": "She apologizes—her orders keep her busy—but she presses a tiny scrap of paper into his palm as a small comfort."
    },
    {
        "neutral": "A narrow lane leads to a creaking library whose windows hold dust-motes and quiet secrets.",
        "happy": "Inside, an old librarian remembers him and hands over a bound folktale, nodding as if passing a torch.",
        "uplift": "A loose page flutters from a book and lands on a sketch of an odd symbol; Arin tucks it away like a promise.",
        "sad": "The librarian closes a heavy volume and tells a gentle story of distant places; Arin listens and feels oddly steadied."
    },
    {
        "neutral": "Along the river stands a smith's workshop with copper bells and the smell of coal and iron.",
        "happy": "The smith shows Arin a small, polished pendant and affixes it to his cord—'For luck,' he says, winking.",
        "uplift": "A bell rings strangely overhead; the smith beckons to the quay and points to a tiny boat with painted stars.",
        "sad": "The smith hums a low tune as he works; Arin watches the steady hammer and feels a quiet warmth in his chest."
    },
    {
        "neutral": "At the hillock he sees a small cottage with a crooked chimney and smoke like memory curling skyward.",
        "happy": "An old friend appears on the porch, arms wide; they trade stories and a flask of warm tea.",
        "uplift": "A child runs out clutching a paper kite that catches the wind and pulls Arin toward a hill where glimmers gather.",
        "sad": "The cottage windows are dim; a neighbor waves sadly but offers a slice of bread and a short tale that comforts."
    },
    {
        "neutral": "Dusk finds him at a lantern-lit bridge where travelers pass with bags and hopeful eyes.",
        "happy": "He helps a tired merchant carry a crate and is rewarded with a soft, grateful laugh and a new companion.",
        "uplift": "Fireflies gather in a drift along the rails—each one a little compass leading him to a strange, bright gate.",
        "sad": "A traveler speaks of loss on quiet nights; Arin listens and offers a steady shoulder before walking on alone."
    },
    {
        "neutral": "Night falls and Arin walks toward a lighthouse on the headland, its beam rolling like a watchful eye.",
        "happy": "At the top he looks out over the sea, laughing as the lights wink back like friends; the world feels wide and kind.",
        "uplift": "An unexpected ship on the horizon flashes a signal; Arin senses that a journey is beginning.",
        "sad": "He watches the tide pull at the shore and remembers things that ache; still, the lighthouse steadies him like a promise."
    }
]
ENDING = "Arin passes through town and beyond—every person met, every door opened, turns into a new story he carries forward."

class StorySystem:
    def __init__(self):
        # scene index points to the current scene to be shown
        self.scene_idx = 0

    # get the current scene's line for the given emotion without advancing
    def get_line(self, emo):
        if self.scene_idx < len(SCENES):
            scene = SCENES[self.scene_idx]
            return scene.get(emo, scene.get("neutral"))
        return ENDING

    # advance to the next scene and return its line for the given emotion
    def advance(self, emo):
        # move forward if possible
        if self.scene_idx < len(SCENES) - 1:
            self.scene_idx += 1
            return self.get_line(emo)
        # if already at last, return ENDING (and keep index at end)
        self.scene_idx = len(SCENES)
        return ENDING

# Ensure the project directory is on sys.path so local imports work regardless of CWD
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera (VideoCapture(0)).")
        return
    det = EmotionDetector()
    # use base_dir for assets path
    assets_walk = os.path.join(base_dir, "assets", "player", "walk")
    player_anim = PlayerAnimation(assets_walk, size=(160,160))  # optional folder
    game = GameWindow(width=960, height=540, player_anim=player_anim)
    story = StorySystem()

    active = False
    last = 0
    current_story = ""
    last_emo = "neutral"
    # keep history of last detected emotions for display
    history = deque(maxlen=30)
    prev_gray = None

    # new: track last chosen emotion at progression and its scores
    last_chosen_emo = None
    last_progression_scores = {}

    while True:
        ret, f = cap.read()
        if not ret or f is None:
            # small wait and continue if frame not available
            time.sleep(0.05)
            continue
        f = cv2.flip(f, 1)
        # motion detection for walking
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        walking = False
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff)
            walking = motion > 7.0  # threshold, tune as needed
        prev_gray = gray

        emo, conf, box, scores = det.detect_stable(f)  # unpack scores
        history.append(emo)

        now = time.time()
        face_present = box is not None

        # Always update the displayed line for the current scene to reflect the live emotion
        if active:
            current_story = story.get_line(emo)
        else:
            current_story = "Press S to start the adventure!"

        # helper: choose which emotion to use when advancing
        def choose_advance_emotion(scores, last_chosen, last_scores):
            if not scores:
                return last_chosen or "neutral"

            # Make a working copy to apply a gentle bias toward happier outcomes
            eff = dict(scores)

            # determine current dominant emotion from raw scores
            raw_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            raw_top, raw_top_score = raw_items[0]

            # If the scene is currently neutral or sad (or last chosen was), steer toward 'happy' (or 'uplift')
            if raw_top in ("neutral", "sad") or (last_chosen in ("neutral", "sad")):
                if "happy" in eff:
                    # add a bonus so 'happy' more readily wins when neutral/sad dominates
                    eff["happy"] = eff.get("happy", 0.0) + max(0.25, raw_top_score * 0.35)
                elif "uplift" in eff:
                    # fallback to uplift if no explicit happy channel
                    eff["uplift"] = eff.get("uplift", 0.0) + max(0.2, raw_top_score * 0.25)

            # Now choose using the biased scores but keep other heuristics
            items = sorted(eff.items(), key=lambda kv: kv[1], reverse=True)
            top, top_score = items[0]
            second = items[1] if len(items) > 1 else (None, 0.0)
            second_name, second_score = second if second is not None else (None, 0.0)

            # If nothing chosen before, take top
            if not last_chosen:
                return top

            # If a different emotion is now top, pick it (it overtook)
            if top != last_chosen:
                return top

            # If top is same as last chosen, see if some emotion surpassed the previous chosen when comparing to last progression
            if last_scores:
                candidates = [(e, s, s - last_scores.get(last_chosen, 0.0)) for e, s in eff.items() if e != last_chosen]
                pos = [c for c in candidates if c[2] > 0]
                if pos:
                    pos.sort(key=lambda t: t[2], reverse=True)
                    return pos[0][0]

            # Otherwise, allow switching to second-highest if it's reasonably close to top
            if second_name and (second_score >= top_score * 0.6 or second_score >= 0.45):
                return second_name

            # fallback to top
            return top

        # Advance story automatically when active, we have a visible face and confidence is reasonable.
        if active and face_present and (now - last > 3.5) and conf > 0.25:
            chosen = choose_advance_emotion(scores, last_chosen_emo, last_progression_scores)
            current_story = story.advance(chosen)
            # record progression choice and scores snapshot
            last_chosen_emo = chosen
            last_progression_scores = dict(scores)
            last = now

        # When emotion changes while active, show a small reaction but do NOT advance scene.
        if active and emo != last_emo and conf > 0.20:
            current_story = f"A shift: you seem {emo}. " + story.get_line(emo)
            last = now

        last_emo = emo

        # Draw animation and story text, pass history for visualization
        out = game.render(f, emo, conf, box, active, current_story, list(history), walking=walking)
        cv2.imshow("AI Director", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            active = True
            current_story = story.get_line("neutral")
            last = now
            last_chosen_emo = None
            last_progression_scores = {}
        if k == ord(" "):
            if active:
                # manual advance uses selection logic as well
                chosen = choose_advance_emotion(scores, last_chosen_emo, last_progression_scores)
                current_story = story.advance(chosen)
                last_chosen_emo = chosen
                last_progression_scores = dict(scores)
                last = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled error:", e)
        raise

