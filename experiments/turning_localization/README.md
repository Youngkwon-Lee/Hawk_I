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

## Results (official test, 8 subjects, 116 clips, 158 labelled far turns)

Trained on the official train split (22 subjects, 310 clips). Settings chosen by 5-fold subject-grouped
cross-validation inside train (40 epochs, threshold 0.6); test evaluated once. This is an **internal
held-out test** (same dataset, site and annotator), not external validation.

**Primary metric: segment F1 at temporal IoU (tIoU) 0.5**, with one-to-one matching. F1 at 0.3/0.7 and
the mean over tIoU 0.1–0.7 are also reported. Produced by `eval_report.py` (set `PD4T_TURN_MODELS` to the checkpoint folder; checkpoints live in the private repo `pd4t-turn-localizer`).

| method | F1@tIoU0.3 | **F1@tIoU0.5** | F1@tIoU0.7 | mean F1 (tIoU 0.1–0.7) | centre error (median) |
|---|---|---|---|---|---|
| BiGRU, mean ± sd of 3 seeds | 0.925 ± 0.014 | **0.861 ± 0.013** | 0.487 ± 0.008 | 0.822 ± 0.011 | 0.11 s |
| Rule: walking-path reversal | 0.679 | 0.455 | 0.167 | 0.516 | 0.18 s |
| Rule: body-height extrema | 0.508 | 0.270 | 0.027 | 0.370 | 0.37 s |
| Trivial: one fixed segment at mean train turn time | 0.007 | 0.000 | 0.000 | 0.007 | – |
| Trivial: whole clip as one segment | 0.000 | 0.000 | 0.000 | 0.006 | – |
| Control: BiGRU trained on shuffled labels | 0.167 | 0.076 | 0.022 | 0.134 | 0.58 s |

- Subject-level bootstrap 95% CI (seed 100, 1000 resamples of the 8 test subjects): F1@tIoU0.5 0.81–0.92.
  Per-subject F1@tIoU0.5 ranges 0.70–0.97.
- Boundaries are the weak point: at tIoU 0.7 F1 drops to ~0.49. Labels are short pivots (~0.9 s on
  average, i.e. ~14 frames at 15 fps), so a few frames of boundary disagreement change the IoU a lot.
- Leakage checks: no subject or visit overlap between train and test; feature normalisation and all
  model selection use train only; the shuffled-label control collapses (F1@tIoU0.5 0.08), so the score
  is not produced by the pipeline itself.

**About the earlier "centre-match F1".** An earlier version of this README (and of this PR) reported F1 0.935 using a
lenient rule (label centre inside a predicted segment ±0.5 s). That rule rewards long segments:
predicting the whole clip already scores 0.774 on it, above both rule baselines (0.746, 0.744). It is
kept in `eval_report.py` only as a secondary metric and should not be used as the headline number.

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
