# MotorExam-VQA Finger Feature-First Plan v0.1

## Summary

For `finger_tapping`, current experiments show:

- VLM `observation-only` improves visibility into qualitative cues but still collapses on clinically useful score separation.
- `score-only` VLM training underperforms the base model.
- `observation + feature fusion` helps only slightly.
- existing kinematic features alone already outperform current VLM-based finger scoring paths.

## Strongest Current Baseline

Using `finger_tapping_*_features_stratified.csv`:

- best model: `extra_regressor_round__shortlist`
- validation:
  - exact: `0.650`
  - MAE: `0.359`
- test:
  - exact: `0.570`
  - MAE: `0.469`

This is currently the most credible finger-tapping score path in the project.

## Recommended Architecture

### Primary

`kinematic features -> score model`

Use:

- `tapping_speed`
- `amplitude_mean`
- `peak_velocity_mean`
- `amplitude_decrement`
- `rhythm_variability`
- `fatigue_rate`
- `amplitude_slope`
- `velocity_decrement`
- `hesitation_count`
- `halt_count`

### Secondary

`VLM observation -> auxiliary explanation only`

Use the VLM for:

- rationale draft
- qualitative review aid
- evidence localization

But do not let it be the main score path unless finger observation quality improves substantially.

## Why This Plan

The current finger observation outputs still collapse:

- many questions resolve to nearly identical answers across scores
- severity prompts collapse toward `mild`
- relative prompts still trend heavily toward `no`

This means stage-1 finger observation is not yet information-rich enough to drive a reliable stage-2 score model.

## What to Improve Next

1. better finger observation prompts
2. stronger weak labels for observation questions
3. feature-informed prompting only as an auxiliary experiment
4. keep the feature-first score model as the practical baseline
