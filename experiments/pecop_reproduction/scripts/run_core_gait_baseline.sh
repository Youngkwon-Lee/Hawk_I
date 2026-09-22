#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_DIR="${CORE_DIR:-$ROOT/_upstream/CoRe}"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0}"
EXP_NAME="${EXP_NAME:-pd4t_gait_core_seed0}"

if [[ -z "${PD4T_ROOT:-}" ]]; then
  echo "PD4T_ROOT is required" >&2
  exit 2
fi

"$ROOT/scripts/preflight.sh"
"$ROOT/scripts/bootstrap_upstreams.sh"
"$PYTHON" "$ROOT/scripts/patch_core_for_pd4t.py" "$CORE_DIR"

if [[ ! -f "$CORE_DIR/model_rgb.pth" ]]; then
  echo "Missing Kinetics I3D weight: $CORE_DIR/model_rgb.pth" >&2
  echo "Place the exact K400 I3D checkpoint referenced by CoRe/PECoP there." >&2
  exit 3
fi

cd "$CORE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
"$PYTHON" main.py --benchmark PD4T --exp_name "$EXP_NAME"
