# MotorExam-VQA Qwen LoRA Execution Plan v0.1

**Updated**: 2026-03-29  
**Author**: Codex (GPT-5)  
**Status**: Draft v0.1

---

## 1. Purpose

This document defines the practical execution order for running the first Qwen LoRA experiments for MotorExam-VQA.

It is intentionally operational rather than conceptual.

The goal is to move from:

- approved / prefill JSONL

to:

- SFT-ready JSONL
- dry-run validation
- LoRA bootstrap
- train-ready environment

---

## 2. Recommended Environment Order

### First choice

- `HPC` or `Runpod Pod`

### Why not local first

- local bootstrap already failed due to disk limits while downloading Qwen weights
- the real target environment for training is GPU-based anyway

---

## 3. Execution Stages

### Stage A: Build Training Data

Generate SFT-format JSONL from MotorExam-VQA records.

Use:

- `scripts/data/export_motorexam_qwen_sft.py`

Expected outputs:

- train SFT JSONL
- validation SFT JSONL

### Stage B: Config Validation

Run:

- `scripts/vlm/train_qwen_vl_lora.py --dry-run`

Purpose:

- check paths
- check task filters
- check SFT record shape

### Stage C: Bootstrap Model

Run:

- `scripts/vlm/train_qwen_vl_lora.py --bootstrap-model`

Purpose:

- verify model loading
- verify processor loading
- verify LoRA attachment

### Stage D: Implement Training Loop

Only after Stage B and Stage C are stable:

- add actual trainer loop
- begin with gait-only SFT

---

## 4. Recommended Initial Data

### Train

- silver SFT JSONL
- gait first

### Validation

- gold SFT JSONL
- small but trusted

### Toy starting point

Current ready toy files:

- `data/processed/motorexam_vqa/sft/motorexam_qwen_sft_silver_batch10_v0_1.jsonl`
- `data/processed/motorexam_vqa/sft/motorexam_qwen_sft_gold_v0_1.jsonl`

---

## 5. Recommended First Run

### Config

- `experiments/configs/vlm/qwen_vl_lora_motorexam_toy_v0_1.yaml`

### Command

```bash
python scripts/vlm/train_qwen_vl_lora.py \
  --config experiments/configs/vlm/qwen_vl_lora_motorexam_toy_v0_1.yaml \
  --dry-run \
  --include-rationale
```

### Next command

```bash
python scripts/vlm/train_qwen_vl_lora.py \
  --config experiments/configs/vlm/qwen_vl_lora_motorexam_toy_v0_1.yaml \
  --dry-run \
  --include-rationale \
  --bootstrap-model
```

---

## 6. Runpod / HPC Practical Checklist

Before bootstrap:

- enough disk for model cache
- enough disk for Hugging Face downloads
- enough VRAM for Qwen2.5-VL-7B + LoRA
- stable network to Hugging Face Hub

Recommended cache paths:

- Hugging Face cache on large attached volume
- project repo separate from model cache

---

## 7. Runpod Workspace Suggestion

Suggested layout:

```text
/workspace
  /hawkeye
  /cache/huggingface
  /results/hawkeye
  /logs/hawkeye
```

Recommended environment variables:

```bash
export HF_HOME=/workspace/cache/huggingface
export TRANSFORMERS_CACHE=/workspace/cache/huggingface
```

---

## 8. Known Current Blocker

The current training entry script can:

- read config
- read SFT JSONL
- summarize splits
- bootstrap LoRA

But:

- the actual training loop is not implemented yet

That is the next engineering step after environment validation.

---

## 9. Immediate Next Engineering Step

Implement a first actual trainer path for:

- gait-only
- score-only or score+observation
- LoRA SFT

Do not start with all four tasks at once.

---

## 10. Decision Summary

Recommended immediate order:

1. export SFT
2. dry-run
3. bootstrap on GPU machine
4. add minimal trainer loop
5. gait-only first experiment

This is the safest path to first useful LoRA results.
