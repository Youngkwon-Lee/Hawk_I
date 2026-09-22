#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-hawkeye-pecop-repro:cuda}"
PD4T_ROOT="${PD4T_ROOT:-}"

if [[ -z "$PD4T_ROOT" ]]; then
  echo "PD4T_ROOT must point to the local PD4T root directory." >&2
  exit 2
fi

if [[ ! -d "$PD4T_ROOT" ]]; then
  echo "PD4T_ROOT does not exist: $PD4T_ROOT" >&2
  exit 2
fi

docker build -f "$ROOT/Dockerfile.cuda" -t "$IMAGE" "$(cd "$ROOT/../.." && pwd)"

docker run --rm -it \
  --gpus all \
  -e PD4T_ROOT=/data/pd4t \
  -v "$PD4T_ROOT:/data/pd4t:ro" \
  -v "$ROOT/results/runtime:/workspace/Hawk_I/experiments/pecop_reproduction/results/runtime" \
  "$IMAGE"
