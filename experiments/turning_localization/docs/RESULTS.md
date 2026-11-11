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

## 3. External (TULIP, 12 subjects, frozen BiGRU)

Turn location/count transfers; boundaries do not (whole far-end dwell of 5–6 s marked). Needs ~60 human
turn labels for a quantitative result (`tulip_external.py`).

## Open items

Second rater (test 116), TULIP labels, more VLM seeds, annotator consent/authorship, venue (EMBC 2027-01-24 /
MIDL abstract 2026-11-30).
