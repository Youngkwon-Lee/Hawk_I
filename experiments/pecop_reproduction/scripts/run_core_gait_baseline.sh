#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_DIR="${CORE_DIR:-$ROOT/_upstream/CoRe}"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0}"
EXP_NAME="${EXP_NAME:-pd4t_gait_core_seed0}"
PD4T_FRAME_ROOT="${PD4T_FRAME_ROOT:-$ROOT/results/runtime/gait_frames}"

if [[ -z "${PD4T_ROOT:-}" ]]; then
  echo "PD4T_ROOT is required" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export ROCR_VISIBLE_DEVICES="$GPU_IDS"
"$PYTHON" - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required; install a CUDA or ROCm/HIP build first") from exc

if not torch.cuda.is_available():
    raise SystemExit("No CUDA/HIP GPU is visible to PyTorch; refusing to start training")

if torch.version.hip:
    backend = f"ROCm/HIP {torch.version.hip}"
elif torch.version.cuda:
    backend = f"CUDA {torch.version.cuda}"
else:
    raise SystemExit("PyTorch reports a GPU but its build is neither ROCm/HIP nor CUDA")

print(f"GPU runtime: {backend}; device: {torch.cuda.get_device_name(0)}")
PY

"$ROOT/scripts/preflight.sh"
"$PYTHON" "$ROOT/scripts/prepare_gait_frame_cache.py" --frame-root "$PD4T_FRAME_ROOT"
export PD4T_FRAME_ROOT
"$ROOT/scripts/bootstrap_upstreams.sh"
"$PYTHON" "$ROOT/scripts/patch_core_for_pd4t.py" "$CORE_DIR"

if [[ ! -f "$CORE_DIR/model_rgb.pth" ]]; then
  echo "Missing Kinetics I3D weight: $CORE_DIR/model_rgb.pth" >&2
  echo "Place the exact K400 I3D checkpoint referenced by CoRe/PECoP there." >&2
  exit 3
fi

cd "$CORE_DIR"
"$PYTHON" main.py --benchmark PD4T --exp_name "$EXP_NAME"
