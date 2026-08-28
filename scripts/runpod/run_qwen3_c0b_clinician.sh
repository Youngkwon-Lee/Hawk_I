#!/usr/bin/env bash
set -euo pipefail

dataset_sha="${HAWKEYE_DATASET_SHA256:?Set HAWKEYE_DATASET_SHA256}"
data_dir="${HAWKEYE_C0B_DATA_DIR:-/workspace/hawkeye-c0b-data}"
run_dir="${HAWKEYE_C0B_RUN_DIR:-/workspace/runs/hawkeye-c0b-clinician-v2-seed42}"
repo_dir="${HAWKEYE_REPO_DIR:-/workspace/Hawk_I}"

cd "$repo_dir"

python scripts/vlm/train_qwen3_c0b_clinician.py validate \
  --data-dir "$data_dir" \
  --expected-dataset-sha256 "$dataset_sha"

python scripts/vlm/train_qwen3_c0b_clinician.py train \
  --data-dir "$data_dir" \
  --expected-dataset-sha256 "$dataset_sha" \
  --output-dir "$run_dir" \
  --candidate-name hawkeye-c0b-clinician-v2-seed42 \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --base-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --seed 42 \
  --fps 5 \
  --frame-width 512 \
  --max-length 12288 \
  --epochs 3 \
  --gradient-accumulation-steps 8 \
  --learning-rate 0.0002 \
  --quantization 4bit

# The training command evaluates validation and writes a candidate manifest.
# Test remains locked until model selection is frozen and explicitly unlocked.
