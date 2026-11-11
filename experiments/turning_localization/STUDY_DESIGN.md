# Study design: far-end turn and freezing localisation in PD gait video (draft, 2026-11-11)

## Positioning against the two reference lines

| | Your Turn (Cheng et al., arXiv 2408.08182) | Video-based FoG detection (e.g. TNSRE 2024 clinical-practice study; graph-based FoG from video) | This study |
|---|---|---|---|
| Task | turning **angle** of clinician-trimmed turn clips (45° bins) | FoG present/absent per clip or window | **when** the far-end turn happens inside an untrimmed 20 s walk (temporal localisation), plus FoG intervals |
| Labels | 2 clinicians, 50% double-rated, kappa 0.96 | clinician FoG ratings | 1 clinician, far-end turnaround only; second rater planned |
| Pose | AlphaPose 2D → Strided Transformer 3D, hip + knee vectors | 2D/3D skeleton or RGB | MediaPipe PoseLandmarker 2D + world |
| Data | Turn-REMAP home (12 PD + 12 controls), Turn-H3.6M | clinic videos | PD4T gait, official 22/8 subject split |

Gap: no prior work localises turns (start/end) in clinical PD gait video against clinician intervals;
Your Turn assumes the turn is already cut out. Turn localisation is the step Your Turn's angle and the
FoG-during-turn analysis need upstream.

## Primary question and metric

- Primary: far-end turn localisation, **F1 at tIoU 0.5** (one-to-one matching), PD4T official test split.
- Secondary: F1 at tIoU 0.3/0.7, mean F1 over tIoU 0.1–0.7, centre error, subject-bootstrap 95% CI.
- Not used as headline: centre-match F1 (whole-clip prediction scores 0.77 on it).

## Comparators (all on the same 116 test clips, same evaluator)

1. Trivial: fixed segment at mean train turn time; whole clip.
2. **Your Turn-definition baseline** (`yourturn_baseline.py`): hip+knee ground-plane yaw,
   turn = >=45° rotation run above a speed threshold, far-end by body size; thresholds tuned on train.
3. Hand rules: body-height extrema; walking-path reversal.
4. Supervised Conv1d-BiGRU on pose (main model).
5. Control: same model trained on shuffled labels.
6. Zero-shot VLM: Qwen3-VL-4B-Instruct temporal grounding (`qwen_turn_zeroshot.py`, scored by `eval_vlm.py`).
7. **Human ceiling (planned):** second clinician re-labels the 116 test clips blind; inter-rater F1@tIoU
   and kappa reported on the same scale as the models.

## Current results (test, 8 subjects, 158 far turns)

| method | F1@tIoU0.3 | F1@tIoU0.5 | F1@tIoU0.7 |
|---|---|---|---|
| BiGRU (3 seeds) | 0.925 | 0.861 ± 0.013 | 0.487 |
| Rule: walking-path reversal | 0.679 | 0.455 | 0.167 |
| Your Turn-definition baseline | 0.546 | 0.283 | 0.082 |
| Rule: body-height extrema | 0.508 | 0.270 | 0.027 |
| Qwen3-VL-4B zero-shot (4 fps, fixed prompt) | 0.270 | 0.161 | 0.029 |
| Qwen3-VL-4B QLoRA fine-tuned (train 310, 1 run) | 0.933 | 0.831 | 0.454 |
| Qwen3-VL-4B zero-shot, prompt v2 (post-hoc) | 0.226 | 0.088 | 0.007 |
| Shuffled-label control | 0.167 | 0.076 | 0.022 |
| Trivial baselines | ≤0.007 | 0 | 0 |

Rotation inside the clinician labels, measured with the Your Turn angle formula: median 128°
(IQR 103–162°). Inside BiGRU predictions: median 155° (IQR 135–188°). So the "pivot" labels span most
of the 180° turnaround, and the model segments slightly wider than the clinician. This links the
localisation output to Your Turn's angle measure and is a candidate clinical readout.

## Freezing (FoG) arm

- Labels: train 16 clips / 6 subjects, test 12 clips / 4 subjects. Too few for interval localisation
  as a primary endpoint. Report as exploratory: clip-level FoG detection and FoG-during-turn
  descriptives, compared with the video FoG literature (clip-level AUC ~0.9 in clinic video).
- Clinical association: turn duration / rotation speed vs MDS-UPDRS 3.10 gait score.

## Validity statements to make in the paper

- Internal held-out test on PD4T (same site, camera, annotator); **not external validation**.
- Leakage controls: subject- and visit-disjoint split, train-only normalisation and model selection,
  shuffled-label control.
- Small test set (8 subjects): report subject-bootstrap CI and per-subject scores.
- Label convention: far-end turnaround only; near-end turns not labelled.
- External validation target: HawkEye hospital videos, if consent and labels allow.

## Open items

- Second rater for the 116 test clips (needed for the human ceiling).
- Agreement from the original annotator on authorship and use of the labels.
- Fine-tuned Qwen3-VL-4B reached F1@tIoU0.5 0.831, not significantly different from the BiGRU (paired
  subject bootstrap CIs include 0). Paper framing: a 74 k-parameter pose model matches a fine-tuned 4 B VLM.
- Optional: more fine-tuning seeds; a larger VLM (8B) if GPU time allows.
- Choose target venue (MIDL abstract 11/30, paper 12/4; EMBC 1/24; CHIL 2/3).
