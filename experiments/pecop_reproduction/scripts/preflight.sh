#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$ROOT/scripts/verify_pd4t_split.py" --self-test
"$PYTHON" "$ROOT/scripts/evaluate_srcc.py" --self-test

"$PYTHON" "$ROOT/scripts/verify_pd4t_split.py" \
  --task Gait \
  --manifest-out "$ROOT/results/runtime/gait_split_manifest.json"

echo "PECoP gait reproduction preflight: PASS"
