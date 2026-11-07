# User study protocol

A short within-subject study of whether the **adaptive** director feels more responsive than two
non-adaptive baselines. Written before any data is collected, so the primary outcome and the
analysis are fixed in advance.

## Hypothesis and design

- **H1 (primary):** participants rate the story as more *responsive* in the **adaptive** condition
  than in the **fixed** condition, and than in the **random** condition.
- **Design:** every participant plays the story three times, once per condition, in a
  counterbalanced order (six possible orders, used equally often). They rate each run straight after.
- **Conditions** (the camera is on in all three, so the experience looks similar):
  - `fixed`: the same route and neutral wording for everyone
  - `random`: random routes and mood variants (reacts to nothing)
  - `adaptive`: the real engine
- **Target sample:** 6 to 12 participants (a multiple of 6 balances the orders exactly). With 6
  participants the smallest possible exact p-value is 0.031; with 8 it is 0.008. Fewer than 6 cannot
  produce a meaningful test, so do not run it with fewer.

## Participants

- Adult classmates or friends who volunteer. No one should feel pressured to take part.
- Not people who have seen the project demo in detail (they would recognise the conditions).
- Check whether your college or project guide requires approval for human-participant work, and get
  it before you start. This protocol is a plan, not an approval.

## Materials

- A laptop with a webcam, the same machine and the same room lighting for everyone.
- A seat about arm's length from the screen, face clearly visible.
- `narrative-eval study-plan --participants N --out study` creates `study/plan.csv` (the exact
  command for every run) and a blank `study/responses.csv`.

## Consent (read aloud or print)

> "This is a short study of an interactive story that uses your webcam. The camera is only used to
> estimate your facial expression while you play. **No video or images are recorded or stored**; the
> program saves only numbers such as a mood label and timings, under an anonymous code like P03.
> You will play three short versions and rate each on five questions. It takes about 10 minutes.
> Taking part is voluntary, you can stop at any time without giving a reason, and you can ask me to
> delete your data afterwards. May I proceed?"

Record that consent was given (a tick next to the participant code is enough; do not write names).

## Procedure (about 10 minutes per participant)

1. Read the consent text. Seat the participant. Do not explain the conditions.
2. Say only: "You will play three versions of a short story. Look at the screen naturally and react
   however you like. Afterwards I will ask a few questions."
3. Run the command for **run 1** from `plan.csv` (it uses `--study`, which hides all diagnostic
   displays). The first run includes a 3-second neutral-face calibration: ask them to relax their
   face. Press **S** to start.
4. When the story ends, hand over the questionnaire and record the answers in `responses.csv` under
   the participant code and condition.
5. Repeat for runs 2 and 3.
6. Ask one open question ("anything you noticed?") and note the answer in the `comment` column.
7. **After** all three runs, explain what the conditions were.

## Questionnaire (1 = strongly disagree, 5 = strongly agree)

| Key | Statement |
|---|---|
| `responsive` (primary) | The story felt responsive to me. |
| `engaged` | I felt engaged while playing. |
| `coherent` | The story felt coherent. |
| `replay` | I would play this again. |
| `reacted` | The story reacted to my expression. |

## Data handling

- Participants are identified only by codes (`P01`, `P02`...). Never write names in any file.
- Session logs (`logs/study/*.jsonl`) contain labels and numbers only, never images.
- `study/` and `logs/` are in `.gitignore`. Commit only aggregate results from `docs/eval/`.
- If a participant asks for deletion, remove their rows and log files.

## Analysis plan (fixed in advance)

```bash
narrative-eval study-analyze study/responses.csv --out docs/eval
```

- **Unit of analysis:** the participant. Only participants with ratings in all three conditions are
  included in the paired tests; report how many that is.
- **Primary outcome:** `responsive`, adaptive vs fixed and adaptive vs random. Test: exact paired
  sign-flip permutation test; Holm correction across these two comparisons.
- **Effect and uncertainty:** mean paired difference with a 95% bootstrap interval, wins / ties /
  losses, and the paired effect size dz.
- **Exploratory:** the four other questions are reported with the same statistics but are not
  corrected across questions and must be labelled exploratory.
- **No peeking and no dropping:** do not add participants until it "becomes significant", and do not
  remove participants because of their ratings. Exclusions are allowed only for technical failures
  (for example the camera never found a face), decided without looking at their ratings.
- Report null or negative results exactly as you would a positive one.

## Reporting checklist

- Number recruited, number analysed, and why anyone was excluded.
- Counterbalancing used and the plan seed.
- The software version (commit hash) and machine.
- The table and chart from `docs/eval/` and the limitations in
  [evaluation.md](evaluation.md#6-threats-to-validity-keep-these-in-the-write-up).
- Optional: `narrative-eval counterfactual logs/study/<session>.jsonl` shows what the other policies
  would have done with one participant's recorded reactions.
