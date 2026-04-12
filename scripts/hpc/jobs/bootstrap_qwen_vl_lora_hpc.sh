#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/hawkeye}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/.hf-cache}"
CONFIG_PATH="${CONFIG_PATH:-experiments/configs/vlm/qwen_vl_lora_motorexam_toy_v0_1.yaml}"

cd "$REPO_ROOT"

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

echo "[bootstrap] repo=$REPO_ROOT"
echo "[bootstrap] hf_cache=$HF_CACHE_DIR"
echo "[bootstrap] config=$CONFIG_PATH"

python scripts/vlm/train_qwen_vl_lora.py \
  --config "$CONFIG_PATH" \
  --dry-run \
  --include-rationale

python scripts/vlm/train_qwen_vl_lora.py \
  --config "$CONFIG_PATH" \
  --dry-run \
  --include-rationale \
  --bootstrap-model
