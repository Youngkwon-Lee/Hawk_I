# MotorExam-VQA Two-Stage Plan v0.1

## Core Finding

Current experiments suggest:

1. `observation-only` supervision improves actual-video VLM outputs.
2. `score-only` supervision is unstable and can underperform the base model.
3. A cheap stage-2 calibrator on top of observation outputs is more promising than direct score prediction.

## Evidence

### Actual-video observation-only

- base: `no`
- adapted observation-only: `yes`
- target: `yes`

### Actual-video score-only

- base: `2`
- adapted score-only: `1`
- target: `3`

### Observation → score bridge

Using paired score/observation examples:

- heuristic task-aware bridge:
  - exact rate: `0.5`
  - MAE: `0.5`
- learned calibrator baseline:
  - exact rate: `0.8`
  - MAE: `0.2`

## Recommended Architecture

### Stage 1: Observation extractor

Train the VLM to predict:

- `answer` for observation prompts
- `motion_cue`
- `body_region`
- `longitudinal_change`
- `evidence_span`

### Stage 2: Task-specific score calibrator

Use a lightweight model that consumes structured observation outputs and predicts the final score.

Examples:

- `gait_observation -> gait_score`
- `finger_tapping_observation -> finger_tapping_score`

## Why Task-Specific

Current data availability:

- `gait` paired train examples: `10`
- `finger_tapping` paired train examples: `0` in the current small train split
- `finger_tapping` paired val examples: `1`

This means:

- `gait` calibrator can be prototyped now
- `finger_tapping` calibrator needs more paired train data

## Near-Term Plan

1. keep training `observation-only` adapters
2. keep direct score prediction as a baseline only
3. use `gait` task-specific calibrator for current small-scale experiments
4. add more paired `finger_tapping` score/observation supervision before fitting a task-specific calibrator
5. postpone `comparison` until comparison-train data exists
