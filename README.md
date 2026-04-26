# Human Gaze During Conversational Interaction With a Robot

<p align="center">
  <img src="figures/setup_session.jpg" width="300" alt="Session" />
  <img src="figures/setup_session_closeup.jpg" width="300" alt="Closeup" />
</p>

## Overview
This repository contains the data, analysis scripts, and write-up for a study on human gaze during conversation with the Furhat humanoid robot. The primary goal is to test whether two well-established gaze patterns from human-to-human conversation still hold when the conversation partner is a humanoid robot:

* **RQ1, Turn-taking regulation.** (a) How often do participants look away from the robot at turn start? (b) How often do they look back at the robot at turn end?
* **RQ2, Listening vs Speaking.** (a) What percentage of time is spent looking at the robot in each role? (b) Do participants look at the robot more while listening than while speaking?

## Dataset
* Five participants (N = 5) completed a ten-minute conversation with the Furhat robot.
* Gaze was recorded with Tobii Pro Glasses 3, a head-mounted eye tracker sampling at approximately 50 Hz.
* The robot was driven by a custom Furhat skill with a Gemini LLM backend ([dara-nn/Furhat-eye-tracking](https://github.com/dara-nn/Furhat-eye-tracking)).
* Each gaze sample carries a recording timestamp, Tobii's eye-movement classification (Fixation / Saccade / EyesNotFound / Unclassified) with its duration, the six AOI hits (Face, Eyes, Mouth, Nose, Body, Outside), the snapshot-mapped (x, y) gaze coordinates, and the filtered pupil diameter.

## Methodology Pipeline
1. **Pre-processing.** From the per-sample TSV exported by Tobii Pro Lab (with the six AOI columns Face, Eyes, Mouth, Nose, Body, Outside), assign each sample an aggregated AOI: *on-robot* if Face, *away* if Body or Outside but not Face. Samples with no AOI hit are dropped as invalid.
2. **Turn events.** Turn-start and turn-end events were extracted from the ElevenLabs transcripts with AI assistance and manual review, and stored in `data/turn_events.csv`. The pre-processing collapses consecutive same-speaker events into full speaker turns.
3. **RQ1, look-away / look-back.** For each turn, scan a ±1 s window around turn start (look-away) and turn end (look-back) using an event-detection rule based on consecutive same-label samples, with sub-200 ms runs dropped as noise.
4. **RQ2, role asymmetry.** Compute the percentage of valid samples spent on-robot during the *listening* and *speaking* roles, per participant.
5. **Random-window baseline.** Re-run the RQ1 detector on 100 randomly placed two-second windows per participant to verify the turn-anchored numbers are not chance-level.
6. **Sub-AOI dwell.** Within on-robot samples, break down dwell time across the Face sub-AOIs (Eyes, Mouth, Nose) and the Body AOI.
7. **Pupil arousal.** Compare mean pupil diameter during speaking versus listening, on-robot versus away.

## Results

**RQ1(a), look-away at turn start.** A look-away occurred in 59.4% (14.5%) of turns, 22.2 pp above the random-window rate of 37.2%.

**RQ1(b), look-back at turn end.** A look-back occurred in 38.4% (22.7%) of turns, indistinguishable from the random-window rate of 40.0%. The pattern did not appear.

<p align="center"><img src="figures/rq1_boxplot.png" width="420" alt="RQ1 box plot" /></p>

**RQ2, Listening vs Speaking.** On-robot time was 74.3% (19.6%) while listening and 43.8% (14.7%) while speaking, a 30.5 pp gap. Listening exceeded speaking for every participant.

<p align="center"><img src="figures/rq2_boxplot.png" width="420" alt="RQ2 box plot" /></p>

**Sub-AOI fixation.** While listening, participants fixated the robot's mouth. While speaking, gaze moved up to the eyes.

<p align="center"><img src="figures/subaoi_dwell_pies.png" width="560" alt="Sub-AOI dwell pies" /></p>

<p align="center">
  <img src="figures/heatmap_listening.png" width="300" alt="Listening heatmap" />
  <img src="figures/heatmap_speaking.png" width="300" alt="Speaking heatmap" />
</p>

**Pupil arousal.** Pupil diameter was larger during participant speech than during listening, possibly reflecting cognitive effort during speech planning.

<p align="center"><img src="figures/pupil_arousal.png" width="500" alt="Pupil arousal" /></p>

## Project Structure

```text
data/
├── gaze-data-trimmed/           # Per-participant Tobii TSVs (analysis columns only)
├── transcript-timestamp-data/   # Raw ElevenLabs transcripts with word-level timestamps
├── transcripts-anonymized/      # Anonymized, turn-aligned transcripts
├── from-glass-data/snap.jpg     # Scene-camera snapshot used to draw the AOIs
└── turn_events.csv              # Turn-start and turn-end timestamps

scripts/
├── step1_preprocess.py          # AOI hit-test, aggregated AOI, turn merge
├── step2_rq1_turn_taking.py     # RQ1 look-away / look-back detector
├── step3_rq3_role_asymmetry.py  # RQ2 listening vs speaking
├── step4_aggregate_and_plot.py  # Per-participant CSV and box plots
├── run_main_rqs.py              # Orchestrator for steps 1–4
├── baseline_random_2s.py        # Random-window baseline for RQ1
├── subaoi_dwell.py              # Sub-AOI dwell percentages
├── subaoi_dwell_pies.py         # Sub-AOI pie charts
├── pupil_arousal.py             # Pupil diameter by role and gaze
├── heatmap.py                   # Fixation heatmaps
└── turn_timeline.py             # Turn-event timeline figure

figures/                          # Generated plots and setup photos
appendices/convo-example.mp4      # Example clip from one session
```

## Full Report
For the full write-up, please refer to [research-report.pdf](research-report.pdf).
