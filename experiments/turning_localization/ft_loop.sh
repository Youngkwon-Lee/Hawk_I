#!/bin/bash
# Auto-resume wrapper for qwen_turn_finetune.py: rerun until <out>/recipe.json exists (max 8 tries).
# usage: QWEN3VL_MODEL=/path/to/Qwen3-VL-4B-Instruct bash ft_loop.sh <train_clips_dir> <train_manifest.json> <out_dir> [gpu]
CLIPS=$1; MAN=$2; OUT=$3; GPU=${4:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
unset PYTORCH_CUDA_ALLOC_CONF   # expandable_segments hit a CUDACachingAllocator INTERNAL ASSERT on WSL
for try in 1 2 3 4 5 6 7 8; do
  [ -f "$OUT/recipe.json" ] && break
  echo "=== try $try $(date)" >> "$OUT.loop.log"
  python qwen_turn_finetune.py --clips "$CLIPS" --manifest "$MAN" --out "$OUT" --gpu "$GPU" --epochs 2 >> "$OUT.log" 2>&1
  echo "exit $? $(date)" >> "$OUT.loop.log"
  sleep 20
done
