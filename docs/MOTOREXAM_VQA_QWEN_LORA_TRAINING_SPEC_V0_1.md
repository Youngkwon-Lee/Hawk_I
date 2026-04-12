# MotorExam-VQA Qwen LoRA Training Spec v0.1

**Updated**: 2026-03-29  
**Author**: Codex (GPT-5)  
**Status**: Draft v0.1

---

## 1. Goal

This document translates the Qwen fine-tuning plan into an implementation-ready training specification.

The immediate target is:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `LoRA / QLoRA`
- `MotorExam-VQA`
- structured JSON outputs

The first concrete objective is to improve:

- gait score boundary calibration
- finger tapping score boundary calibration

before scaling to the full four-task benchmark.

---

## 2. Initial Scope

### Phase 1 Tasks

- `gait`
- `finger_tapping`

### Phase 1 Outputs

- `score`
- `observation-style cues`
- `evidence support`

### Why this scope

- strongest current signal
- easiest to compare against existing prompting experiments
- lowest risk before adding hand movement and leg agility

---

## 3. Model

Base model:

- `Qwen/Qwen2.5-VL-7B-Instruct`

Adaptation:

- `LoRA`

Optional later:

- `QLoRA`

Do not begin with:

- full fine-tuning
- custom head-only training

---

## 4. Input Format

### Preferred default

- sampled images, not full raw video tensors

Use the current task-aware policy:

- `gait = 16 frames`
- `finger_tapping = 24 frames`

Reason:

- stable
- already tested in the current Runpod path
- easier to batch than raw full-video training

---

## 5. Output Format

Use structured JSON targets.

### Example target

```json
{
  "answer": 1,
  "visibility": "good",
  "uncertainty_flag": "none",
  "motion_cue": ["reduced_arm_swing"],
  "body_region": ["bilateral_upper"],
  "longitudinal_change": "absent",
  "evidence_span": {"start_sec": 2.0, "end_sec": 7.5}
}
```

### Why JSON

- consistent with current MotorExam-VQA pipeline
- easy to validate
- supports score + observation + evidence together

---

## 6. Training Data Mix

### Train

- mostly `silver`
- selected `gold` if available

### Validation

- `gold`

### Test

- `gold`

### Recommended first mix

- `80-90% silver`
- `10-20% gold/calibration`

---

## 7. Training Example Construction

### Input

- task instruction
- scoring rubric
- task-specific guidance
- sampled frames

### Target

- structured JSON answer

### Prompt shape

```text
You are reviewing a Parkinson's motor assessment video.
Task: gait
Question: What is the gait score for this trial?
Scoring rubric: ...
Task-specific guidance: ...
Return JSON only.
```

---

## 8. LoRA Target Modules

Recommended first attempt:

- attention projection layers
- MLP projection layers
- multimodal connector / projector if exposed in the implementation

Suggested initial LoRA hyperparameters:

- `r = 16`
- `alpha = 32`
- `dropout = 0.05`

If unstable:

- reduce `r` to `8`

---

## 9. Loss Strategy

### Option A: pure SFT

Train the model to emit the structured JSON target directly.

### Option B: weighted JSON training

Still emit JSON, but oversample or weight:

- low-score / near-normal examples
- score-boundary examples
- disagreement-heavy examples

Recommended first step:

- start with plain SFT
- add weighting later if collapse persists

---

## 10. Curriculum Recommendation

### Step 1

Train only `gait score`

### Step 2

Train `gait score + gait observations`

### Step 3

Add `finger_tapping`

### Step 4

Add evidence span/cue fields

This staged curriculum is preferred over full multi-task training from day one.

---

## 11. Sampling Ablation Plan

For the first LoRA experiments, compare:

- `uniform-8`
- `task-aware gait-16 / finger-24`

Do not start with:

- dense 400-frame sampling
- all-frame decoding

Reason:

- higher cost
- no clear quality gain from current experiments

---

## 12. Evaluation Plan

### Main metrics

- `QWK`
- `MAE`
- `Spearman`

### Secondary metrics

- cue agreement
- span validity
- JSON validity rate

### Mandatory baselines

- zero-shot Qwen
- calibrated-prompt Qwen
- LoRA Qwen

---

## 13. Minimum Viable Experiment

First experiment should be:

- task: `gait`
- input: task-aware `16` frames
- target: `score` only
- training style: `SFT JSON`
- validation: gold gait score set

Success criterion:

- lower MAE than zero-shot
- less collapse toward one middle score

---

## 14. Second Experiment

After the first MVP:

- add `finger_tapping`
- keep sampled-frame input
- add `motion_cue`
- add `longitudinal_change`

Goal:

- teach the model not only the final answer
- but also the intermediate clinical reasoning structure

---

## 15. Key Risks

### Risk 1: Score collapse persists

Mitigation:

- add class balancing
- add more gold anchors
- add explicit observation supervision

### Risk 2: JSON drift

Mitigation:

- strict output format
- JSON validator in evaluation loop

### Risk 3: Overfitting to silver noise

Mitigation:

- keep gold validation separate
- inspect disagreement buckets

---

## 16. Decision Summary

Recommended immediate implementation:

- `Qwen2.5-VL-7B-Instruct`
- `LoRA`
- `gait first`
- `task-aware sampled frames`
- `structured JSON SFT`

This is the smallest serious training experiment that can answer whether fine-tuning materially improves MotorExam-VQA.
