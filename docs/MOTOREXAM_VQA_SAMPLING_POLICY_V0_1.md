# MotorExam-VQA Sampling Policy v0.1

**Updated**: 2026-03-29  
**Author**: Codex (GPT-5)  
**Status**: Draft v0.1

---

## 1. Purpose

This document defines a practical sampling strategy for **MotorExam-VQA** when using multimodal VLMs such as `Qwen/Qwen2.5-VL-7B-Instruct`.

The goal is not to feed every frame. The goal is to preserve the clinical cues that matter for `MDS-UPDRS` scoring while keeping inference stable and affordable.

This policy is motivated by:

- MDS-UPDRS scoring logic
- long-video VLM research trends
- Hawkeye's own Runpod experiments

---

## 2. Key Observation

In current Hawkeye experiments:

- very dense uniform sampling does **not** automatically improve quality
- `400` sampled frames from a single gait video produced a poor clinical answer
- shorter raw clips can run successfully, but raw full-length videos are less stable

Therefore:

- `all-frame` or `near-all-frame` inference is **not** the recommended default
- task-aware sampling is preferred over uniform dense sampling

---

## 3. Research Direction Summary

Recent long-video VLM work generally favors:

- frame selection
- adaptive compression
- hierarchical context
- query-conditioned sampling

Representative directions:

- `LongVU`: spatiotemporal adaptive compression
- `LongVA`: long-context video understanding
- `Generative Frame Sampler`: learned frame selection for long video QA

These trends support the same conclusion for MotorExam-VQA:

> clinical video understanding should prioritize **informative moments**, not all frames equally

---

## 4. MDS-UPDRS Principle

For MotorExam-VQA, frame sampling should reflect **what a trained rater actually watches for**.

This means the sampling policy should optimize for:

- progression over time
- rhythm regularity / irregularity
- amplitude decrement
- slowness / hesitation
- asymmetry
- task-specific kinematic cues

Not every frame contributes equally to these decisions.

---

## 5. Global Strategy

Use a three-layer strategy:

### Layer A: Sparse Global Coverage

Purpose:

- preserve whole-trial context
- avoid missing start / middle / end progression

Method:

- sample uniformly across the entire clip

### Layer B: Task-Aware Dense Sampling

Purpose:

- focus on clinically informative motion phases

Method:

- add extra samples around task-specific cue windows

### Layer C: Longitudinal Anchors

Purpose:

- support decrement / fatigue / worsening judgments

Method:

- always include early, middle, and late anchors

---

## 6. Gait Policy

### 6.1 Clinical Cues to Preserve

For `MDS-UPDRS gait`, the most important cues include:

- gait speed
- step length
- reduced arm swing
- trunk / axial posture
- asymmetry
- turning difficulty
- freezing-like hesitation

### 6.2 Recommended Sampling Units

Prioritize:

- gait initiation
- early steady walking
- mid-trial steady walking
- pre-turn segment
- turning segment
- post-turn recovery
- final walking segment

### 6.3 Default Gait Sampling Policy

Recommended default:

- `12-16` sparse global frames
- `8-12` task-aware additional frames around:
  - arm swing visibility
  - turning
  - stride transitions

Total:

- `20-28` effective frames for gait

### 6.4 Gait Longitudinal Anchors

Always include:

- first 15% of clip
- middle 50%
- last 15%

Reason:

- to compare whether gait worsens or remains stable

### 6.5 Gait Side-View Preference

When multiple views exist, prioritize the view that best preserves:

- arm swing
- stride length
- trunk lean

If front-facing only:

- increase emphasis on symmetry and step timing

---

## 7. Finger Tapping Policy

### 7.1 Clinical Cues to Preserve

For `MDS-UPDRS finger tapping`, the most important cues include:

- opening / closing speed
- amplitude decrement
- rhythm irregularity
- hesitation
- interruptions
- fatigue over time

### 7.2 Why Uniform Sampling Alone Is Weak

Finger tapping is cyclic.

A simple uniform sample across the clip may miss:

- incomplete taps
- phase transitions
- brief hesitations
- decrement across repetitions

### 7.3 Default Finger Tapping Sampling Policy

Recommended default:

- `8-10` global frames across the full trial
- plus `12-20` cycle-sensitive frames sampled from:
  - early tapping repetitions
  - mid-trial repetitions
  - late repetitions

Total:

- `20-30` effective frames for finger tapping

### 7.4 Temporal Coverage Rule

Always preserve:

- first repetitions
- middle repetitions
- last repetitions

Reason:

- amplitude decrement is a longitudinal cue, not just a per-frame cue

### 7.5 Finger Region Importance

Sampling and cropping should preserve:

- fingertips
- MCP/PIP movement amplitude
- thumb-index distance or contact pattern when visible

If frame budget is limited, prefer tighter hand visibility over extra whole-body context.

---

## 8. Recommended Default Budgets

### Baseline Production Budget

- `gait`: `16` frames
- `finger_tapping`: `24` frames

### Research Budget

- `gait`: `24` frames
- `finger_tapping`: `32` frames

### Not Recommended as Default

- `400` dense uniform frames
- full-frame, all-frame inference

Reason:

- too slow
- does not reliably improve answer quality
- wastes context on redundant motion

---

## 9. Feature Support for Sampling

Sampling should not be purely visual-uniform.

Use lightweight motion cues to guide frame selection:

### Gait Support Features

- optical flow magnitude
- step periodicity estimate
- arm swing amplitude estimate
- trunk sway estimate
- turn detector

### Finger Tapping Support Features

- fingertip distance / contact cycles
- movement amplitude envelope
- tapping frequency estimate
- interruption detector
- velocity peaks

These can be used **before** VLM inference to choose better frames.

---

## 10. Compression Strategy

### Recommended Compression Types

- temporal uniform downsampling
- event-aware oversampling
- crop-aware spatial compression
- task-conditioned frame budget

### Not Recommended

- naïve full-video frame expansion
- fixed ultra-dense sampling for every task

---

## 11. Proposed Adaptive Policy

### Policy A: Query-Agnostic Task Prior

Select frames using only task knowledge.

Examples:

- gait: emphasize turn + arm swing visibility
- finger tapping: emphasize repetition phases

### Policy B: Query-Conditioned Refinement

After the question is known, refine the sample set.

Examples:

- if the question is about `amplitude decrement`, emphasize early vs late frames
- if the question is about `turning difficulty`, emphasize turn window
- if the question is about `reduced arm swing`, emphasize frames with full bilateral arm visibility

This is the recommended long-term policy.

---

## 12. Hawkeye Recommendation

For current Hawkeye experiments:

### Immediate Default

- `gait`: `16` frames
- `finger_tapping`: `24` frames

### Immediate Native-Video Rule

If raw video is used:

- do **server-side adaptive frame extraction**
- do **not** attempt all-frame inference

### Near-Term Experiment Plan

1. Compare `gait 16 vs 24`
2. Compare `finger tapping 24 vs 32`
3. Add early/middle/late anchor constraints
4. Add task-conditioned frame selector

---

## 13. Decision Summary

### What to Avoid

- all-frame inference
- very dense uniform sampling by default
- assuming more frames always helps

### What to Prefer

- MDS-UPDRS cue-aware sampling
- longitudinal anchors
- task-conditioned frame budgets
- adaptive compression

### Recommended Working Default

- `gait = 16`
- `finger_tapping = 24`

This is the best current balance of:

- clinical relevance
- inference stability
- latency
- cost
