# MotorExam-VQA Qwen Comparison v0.1

**Updated**: 2026-03-30  
**Author**: Codex (GPT-5)  
**Status**: Working comparison snapshot

---

## 1. Scope

This document compares the currently observed behavior of:

- zero-shot Qwen VL
- prompt-calibrated Qwen VL
- small QLoRA fine-tuning run
- overnight QLoRA run

This is not yet a full benchmark paper table.

It is a practical engineering comparison based on completed Hawkeye experiments.

---

## 2. Zero-Shot Raw-Video Gait Probe

### Setup

- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- input: `10-second raw gait clips`
- task: gait scoring probe
- ground-truth scores tested: `0 / 1 / 2 / 3`

### Predictions

- GT `0` -> Pred `3`
- GT `1` -> Pred `2`
- GT `2` -> Pred `2`
- GT `3` -> Pred `2`

### Summary

- exact match: `1 / 4`
- exact rate: `0.25`
- MAE: `1.25`

### Interpretation

The base model can describe motion, but score boundaries are poorly calibrated.

Failure pattern:

- collapse toward the middle score
- overcalling visible abnormality

---

## 3. Prompt Calibration

### Change

Added stricter score rubric and task-specific conservative guidance.

### Result

For near-normal gait / finger tapping toy examples:

- previous output tended to be `2`
- calibrated prompt often reduced that to `1`

### Interpretation

Prompt calibration improves overcalling slightly, but does **not** solve score collapse reliably.

It is helpful, but insufficient as the final solution.

---

## 4. Small QLoRA Run

### Setup

- config: `qwen_vl_lora_motorexam_small_v0_1`
- quantization: `4bit`
- tasks: `gait + finger_tapping`
- train rows: `10`
- val rows: `3`
- sampling policy: `8 / 8`

### Observed Training Log

- loss step 1: `2.33`
- loss step 2: `2.20`
- final train loss: `2.265`
- runtime: `~38.9s`

### Output Artifacts

- `adapter_config.json`
- `adapter_model.safetensors`
- `training_args.bin`

Adapter size:

- approximately `190 MB`

### Interpretation

This confirms that:

- QLoRA training works end-to-end
- adapter saving works
- the current data/collator/trainer path is functional

This is the first stable trainable baseline.

---

## 5. Overnight QLoRA Run

### Setup

- config: `qwen_vl_lora_motorexam_overnight_v0_1`
- quantization: `4bit`
- same base task family
- longer training horizon

### Observed Signals from Logs

The run completed successfully and the logs show sustained low training loss in later epochs.

Examples observed in the late-stage log tail:

- epoch `4.4` -> loss `0.04296`
- epoch `5.0` -> loss `0.0238`
- epoch `8.0` -> loss `0.01731`
- epoch `10.0` -> loss `0.03124`
- epoch `13.0` -> loss `0.02453`

### Output Artifacts

- `adapter_config.json`
- `adapter_model.safetensors`
- `training_args.bin`
- `README.md`

Adapter size:

- approximately `190 MB`

### Interpretation

The overnight run clearly fit the tiny training set much more strongly than the small run.

However:

- this is still training loss
- it does **not** yet prove better generalization

So the correct conclusion is:

> the overnight run trained more thoroughly, but we still need explicit post-train evaluation against the baseline

---

## 6. Practical Comparison Table

| Setting | Status | Key result | Main limitation |
|---|---|---|---|
| Zero-shot Qwen VL | works | sees motion, but MAE `1.25` on 4-score gait probe | score collapse / overcalling |
| Calibrated prompt | works | reduces some overcalling | still collapses toward middle |
| Small QLoRA | succeeds | end-to-end train + adapter save | too small to claim generalization |
| Overnight QLoRA | succeeds | much lower late-stage training loss | evaluation not yet run |

---

## 7. Current Best Interpretation

### What we know

- infrastructure works
- Qwen VL works
- fine-tuning works
- QLoRA on 4090-class Runpod is feasible

### What we do not know yet

- whether the adapter actually beats zero-shot on held-out clinical samples

That is now the most important next question.

---

## 8. Recommended Next Step

Run a direct post-train comparison:

1. base Qwen
2. small adapter
3. overnight adapter

on the same held-out examples.

Primary focus:

- score accuracy
- MAE
- whether score collapse is reduced

---

## 9. Working Conclusion

At the current stage:

- zero-shot is not clinically calibrated enough
- prompt tuning helps only a little
- QLoRA training is the first approach that looks worth pushing further

But the next decision should be made using **post-train evaluation**, not training loss alone.
