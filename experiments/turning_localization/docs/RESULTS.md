# Results summary (2026-11-12)

PD4T official split (train 22 subjects / 310 clips, test 8 / 116). Research use, not clinically validated.
95% CIs: subject-level bootstrap over the 8 test subjects. Scripts in `training/`; outputs listed in
`docs/DATA_LOCATIONS.md`.

## 1. Far-end turn localisation (primary: F1@tIoU0.5)

| method | input | tIoU0.3 | tIoU0.5 | tIoU0.7 | script |
|---|---|---|---|---|---|
| Conv1d-BiGRU, 3 seeds | pose | 0.925 | **0.861 ± 0.013** (CI 0.81–0.92) | 0.487 | `turn_sup.py`, `eval_report.py` |
| Qwen3-VL-4B QLoRA, 1 run | RGB | 0.933 | 0.831 (CI 0.75–0.90) | 0.454 | `qwen_turn_finetune.py` |
| rule: walking-path reversal | pose | 0.679 | 0.455 | 0.167 | `rule_path.py` |
| Your Turn definition (≥45°) | pose | 0.546 | 0.283 | 0.082 | `yourturn_baseline.py` |
| Qwen3-VL-4B zero-shot v1 / v2 | RGB | 0.270 / 0.226 | 0.161 / 0.088 | 0.029 / 0.007 | `qwen_turn_zeroshot.py` |
| shuffled-label control | pose | 0.167 | 0.076 | 0.022 | `eval_report.py` |

BiGRU vs fine-tuned VLM: paired bootstrap, all CIs include 0 (`compare_paired.py`).
Turn duration (matched turns): BiGRU MAE 0.26 s, ICC 0.70; VLM MAE 0.20 s, ICC 0.78 (`duration_agreement.py`).

## 2. UPDRS 3.10 from primitives (target: PD4T official score)

| input | acc | MAE | QWK (CI) | script |
|---|---|---|---|---|
| clinician 8 primitives (oracle upper bound) | 0.71 | 0.31 | 0.746 (0.60–0.84) | `updrs_from_primitives.py` |
| automatic turn timing (5 features) | 0.51 | 0.51 | 0.663 (0.46–0.76) | `updrs_from_turns.py` |
| Qwen3-VL C0B (full RGB, PR #37) | 0.64 | 0.40 | 0.621 (0.46–0.76) | `fusion_c0b.py` |
| C0B + turn/pose late fusion 50:50 | 0.63 | 0.38 | 0.669 (0.53–0.76) | `fusion_c0b.py` |
| annotator's own 3.10 vs official | 0.76 | 0.25 | 0.846 | `updrs_from_primitives.py` |

Caveat: primitives and the annotator's score come from the same rater; the oracle row is optimistic.

## 2b. Automatic primitive bottleneck (pose -> 7 primitives -> UPDRS 3.10), `primitive_bottleneck.py`

Per-primitive test agreement (QWK vs clinician): gait speed reduction 0.89 · shortened stride 0.85 ·
festination 0.76 · postural instability 0.54 · step-length asymmetry 0.41 · trunk flexion 0.29 ·
arm-swing asymmetry 0.16 (freezing excluded).

| route | acc | MAE | QWK (CI) |
|---|---|---|---|
| bottleneck, integrator trained on out-of-fold predicted primitives | 0.66 | 0.37 | **0.743** (0.54–0.87) |
| bottleneck, integrator trained on clinician primitives | 0.61 | 0.43 | 0.679 (0.40–0.84) |
| same pose features, direct to score (no bottleneck) | 0.65 | 0.37 | 0.718 (0.57–0.82) |

Fully automatic; matches the clinician-primitive upper bound (0.746) and is above the RGB VLM C0B (0.621);
differences are within CIs (8 test subjects). One run of a deterministic pipeline.

## 2c. RGB+pose early fusion for the 4 weak primitives (Qwen3-VL-4B QLoRA), `primitive_fusion_eval.py`

8 frames (≤448 px) + the pose measurements as text → JSON scores for arm-swing asymmetry, trunk flexion,
step-length asymmetry, postural instability. Final adapter on all train clips (test predictions) and two
fold2 adapters (train out-of-fold). Clips with a missing rating excluded from each primitive's QWK.

| primitive | pose QWK train-OOF / test | RGB+pose QWK train-OOF / test | RGB+pose non-zero predictions (test) |
|---|---|---|---|
| arm-swing asymmetry | 0.26 / 0.16 | 0.01 / 0.13 | 6 of 115 (clinician 56) |
| trunk flexion | 0.65 / 0.29 | 0.24 / 0.45 | 11 of 114 (clinician 20) |
| step-length asymmetry | 0.34 / 0.41 | 0.00 / 0.00 | 0 of 115 (clinician 29) |
| postural instability | 0.51 / 0.54 | 0.10 / 0.07 | 2 of 114 (clinician 30) |

| UPDRS 3.10 route | acc | MAE | QWK (CI) | Δ vs pose (CI) |
|---|---|---|---|---|
| pose bottleneck (2b, reproduced) | 0.66 | 0.37 | 0.743 (0.54–0.87) | — |
| 4 primitives from RGB+pose | 0.51 | 0.58 | 0.592 (0.30–0.78) | −0.31 to −0.04 |
| per-primitive source chosen on train-OOF | = pose bottleneck (pose chosen for all 4) | | 0.743 | 0 |

Negative result: the fine-tuned VLM collapses to the majority score 0 (no class weighting; missing ratings
trained as 0). The trunk-flexion test gain (0.29→0.45) is not supported by train-OOF (0.24 vs 0.65) and is
treated as chance. Single run; GPU shared, see the run logs.

## 3. External (TULIP, 12 subjects: 8 PD / 4 healthy, Camera1 frontal gait, 3 clinicians)

Turns (frozen BiGRU, `tulip_external.py`): location/count transfer; boundaries do not (whole far-end dwell of
5–6 s marked); quantitative turn agreement needs ~60 human turn labels.

UPDRS 3.10 (`tulip_bottleneck.py`): 60-s video → three 20-s windows → bottleneck rebuilt on PD4T (reproduces
0.743) → mean expected score per subject; no TULIP label is used for fitting.

| variant | Spearman vs clinician mean (p) | MAE | QWK vs rounded mean | exact / within 1 | PD vs HT mean pred (p) |
|---|---|---|---|---|---|
| A frozen (as trained) | 0.29 (0.36) | 0.86 | 0.09 | 33% / 75% | 1.89 vs 1.61 (0.28) |
| B label-free feature alignment (TULIP feature stats → PD4T train mean/sd) | **0.66 (0.02)** | **0.33** | **0.70** | 67% / 100% | 0.86 vs 0.35 (0.06) |
| T turn timing only (reference) | 0.18 (0.58) | 0.53 | 0.19 | 42% / 83% | 1.58 vs 1.15 (0.57) |

Clinicians agree with each other at pairwise QWK 0.46–0.52. Frozen transfer over-scores almost everyone
(~2; predicted festination and postural instability inflated) → camera geometry shifts the feature ranges.
Alignment uses TULIP's own feature distribution (assumes a comparable case mix) — a site-level, label-free
calibration that must be stated as such. n = 12; TULIP pose is 16 fps vs 15 fps in PD4T.

## Open items

Second rater (test 116), TULIP turn labels, VLM primitive re-run with class weighting (or drop), more VLM seeds, annotator consent/authorship, venue (EMBC 2027-01-24 /
MIDL abstract 2026-11-30).
