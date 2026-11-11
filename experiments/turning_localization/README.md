# PD4T gait — far-end turn localisation (research, 2026-11-11)

Pose-based localisation of the clinician-labelled **far-end turnaround pivot moments** in PD4T gait clips.
Code only. No video, pose arrays, labels or per-clip predictions are stored in Git.

## Data and label definition

- Input: the 426 PD4T gait **20-second clips** (`D:/clip/gait`). 419 clips are the first 0–20 s of the
  original video; 7 long videos used a manually selected walking bout whose offset was not recorded.
  Labels were made on these clips, so the clips (not the originals) are the input.
- Labels: single clinician annotator (dongsung). `turning_intervals` mark **only the far-end turnaround,
  only the pivot moment** (typically 0.3–0.6 s for mild cases, longer for severe cases). Near-end turns are
  intentionally not labelled (definition confirmed by the project owner, 2026-11-11).
  `freezing_intervals` are kept but are rare (train 16 clips / 6 subjects, test 12 clips / 4 subjects).
- Split: official PD4T `Annotations/Gait/train.csv` (22 subjects, 310 clips) and `test.csv`
  (8 subjects, 116 clips, 158 labelled far turns). See `docs/research/PD4T_OFFICIAL_SPLIT_AND_APPROVALS_20261110.md`.

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `extract_pose.py` | MediaPipe PoseLandmarker (full, float16) at 15 fps, largest person per frame, saves `npz/` |
| 2a | `rule_height.py` | rule v1: extrema of image body height (far end = smallest) |
| 2b | `rule_path.py` | rule v2: reversals of the walking-progress axis (PCA of hip x and log body height) |
| 3 | `turn_sup.py` | supervised frame classifier (Conv1d + BiGRU, ~74k parameters, trained from scratch on CPU) |
| QA | `contact_sheet.py` | 0.5 s contact sheets with labels and predictions for visual review |

Features for step 3 (76 per frame): 15 joints centred on the hip and scaled by torso length, nose/ear
visibility, log body height, hip x, shoulder-order facing cue, world-space shoulder yaw (sin/cos), and
their first differences. Frame target = 1 inside a label interval; class-weighted BCE.

Model selection uses only the 22 train subjects (5-fold subject-grouped CV over epochs {20, 40} and
threshold {0.3–0.7}); selected 40 epochs, threshold 0.6 (CV F1 0.908). Rule thresholds were also chosen on
train. The test subjects are evaluated once, with 3 training seeds.

Locations are set with environment variables (`config.py`): `PD4T_TURN_ROOT`, `PD4T_CLIP_ROOT`,
`PD4T_TURN_LABEL_DIR`, `PD4T_OFFICIAL_ANN`.

```bash
python extract_pose.py        # ~25 min on 4 CPU cores for 426 clips
python rule_height.py
python rule_path.py
python turn_sup.py            # ~12 min on CPU: 5-fold CV + 3 test seeds + rule comparison
```

## Results (official test, 8 subjects, 158 labelled far turns)

Matching: a label is detected if its centre falls inside a predicted interval ±0.5 s (one-to-one).
tIoU columns use one-to-one interval overlap.

| method | recall | precision | F1 | F1@tIoU0.3 | F1@tIoU0.5 | centre error (median) |
|---|---|---|---|---|---|---|
| rule v1 body height | 0.873 | 0.651 | 0.746 | 0.508 | 0.270 | 0.37 s |
| rule v2 path reversal | 0.734 | 0.753 | 0.744 | 0.679 | 0.455 | 0.18 s |
| Conv1d-BiGRU (mean ± sd, 3 seeds) | 0.956 ± 0.005 | 0.915 ± 0.016 | 0.935 ± 0.009 | 0.925 ± 0.014 | 0.861 ± 0.013 | 0.11 s |

## Limits

- The model reproduces one annotator's convention (far-end pivot only). It is not a general "all turns,
  full duration" detector, and inter-rater agreement has not been measured yet, so no human-level claim.
- Precision counts unlabelled near-end turns as errors only where the detector calls them far-end.
- MediaPipe failures (lowest clip 77 % frames detected) and assistants in frame are not separately handled.
- Research only; no clinical use.

## Provenance

The scripts were written and run by an AI coding assistant (Claude) under the project owner's direction on
the owner's home desktop. The turn detector itself is the small Conv1d-BiGRU above; no large language
model is part of the detection pipeline.
