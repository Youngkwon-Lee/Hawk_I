# MotorExam-VQA Split Experiment Plan v0.1

## Goal

Separate `score`, `observation`, and `comparison` question types so we can see whether Qwen LoRA is failing because:

1. ordinal calibration is hard
2. observation extraction is easier and should be learned first
3. comparison prompts need separate supervision and output constraints

## Current Evidence

From `actual-video` probes:

- `observation` improved with the overnight adapter
- `score` remained unstable
- `comparison` remained structurally weak

This suggests we should not treat all question types as one homogeneous task.

## Current Small-Split Availability

Using `experiments/data/motorexam_qwen_sft_small_train_v0_1.jsonl`:

- train rows: `20`
- train score rows: `10`
- train observation rows: `10`
- train comparison rows: `0`

Using `experiments/data/motorexam_qwen_sft_small_val_v0_1.jsonl`:

- val score rows: `1`
- val observation rows: `1`
- val comparison rows: `1`

## Recommended Order

1. `observation-only`
   - best early target
   - binary presence/absence is easiest to stabilize
2. `score-only`
   - second target
   - isolate ordinal boundary learning from cue extraction
3. `comparison-only`
   - postpone until comparison training examples exist

## Configs

- `experiments/configs/vlm/qwen_vl_lora_motorexam_observation_only_v0_1.yaml`
- `experiments/configs/vlm/qwen_vl_lora_motorexam_score_only_v0_1.yaml`
- `experiments/configs/vlm/qwen_vl_lora_motorexam_comparison_only_v0_1.yaml`

## Expected Outcome

- If `observation-only` improves clearly, the model can perceive motor cues but struggles to map them to UPDRS scores.
- If `score-only` still collapses, we likely need stronger calibration supervision or a two-stage pipeline.
- If `comparison-only` fails after data is added, prompt/schema design may be the main issue rather than representation quality.
