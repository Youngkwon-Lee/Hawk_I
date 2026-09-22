# PECoP / PD4T Gait Reproduction

## Goal

Reproduce the PD4T **Gait** rows from PECoP Table 4 before comparing Hawkeye against the published baseline.

Target SRCC values from the WACV 2024 paper:

| Method | Gait SRCC |
| --- | ---: |
| USDL | 79.14 |
| USDL + PECoP | 80.68 |
| CoRe | 78.87 |
| CoRe + PECoP | 82.33 |

Primary milestone: reproduce **CoRe -> CoRe + PECoP** on the original PD4T Gait train/test split.

## Scope boundary

This directory is a **paper-reproduction exception** to the repository's normal stratified-split policy.

- Reproduction uses the original `Annotations/Gait/train.csv` and `test.csv` only.
- Do **not** use these files for Hawkeye production training.
- Do **not** modify the production stratified split.
- Do **not** commit PD4T videos, annotations, subject IDs, trained weights, or derived row-level labels.
- Default manifests expose only counts, CSV checksums, and hashed subject-set digests.

## Why a separate harness is needed

The upstream PECoP repository contains the continual-pretraining implementation, but its README points to the external CoRe and USDL repositories for downstream AQA fine-tuning/evaluation. Therefore reproduction has two independently checkable stages:

1. downstream baseline reproduction (CoRe / USDL)
2. PECoP continual pretraining + the same downstream baseline

Do not interpret a downstream mismatch as a PECoP failure until the no-PECoP baseline reproduces first.

## Critical split parser rule

PD4T annotation IDs encode the subject as the **final 3-digit suffix**.

- `15-005087_l_042` -> subject `042`
- `15-001760_009` -> subject `009`

Older Hawkeye split scripts use a regex that can interpret the middle numeric token as a patient identifier. This reproduction harness intentionally does not reuse that parser.


## Recommended execution target

Use a supported **Linux + NVIDIA CUDA** GPU for full reproduction. The harness is packaged with `Dockerfile.cuda` so the same environment can run on a V100/A100/RTX-class machine or a cloud GPU.

The current CoRe adapter uses a local-only deterministic frame cache:

1. validate the original Gait split
2. uniformly sample 103 frames from each of the 426 Gait videos once
3. store those derived frames under `results/runtime/gait_frames/` (Git ignored)
4. train CoRe from the cached frames

The 103-frame full-video uniform sampling policy is a **reproduction hypothesis**, not a published PECoP detail. The PECoP paper publishes the 32-frame SSL pretraining setup but states only that downstream baselines follow their original training/evaluation strategies.

Typical CUDA-host flow:

```bash
export PD4T_ROOT=/secure/path/to/PD4T/PD4T/PD4T
bash experiments/pecop_reproduction/scripts/run_cuda_container.sh

# inside the container
bash experiments/pecop_reproduction/scripts/run_core_gait_baseline.sh
```

## Phase 0 — deterministic checks

```bash
python experiments/pecop_reproduction/scripts/verify_pd4t_split.py --self-test
python experiments/pecop_reproduction/scripts/evaluate_srcc.py --self-test

python experiments/pecop_reproduction/scripts/verify_pd4t_split.py \
  --task Gait \
  --manifest-out experiments/pecop_reproduction/results/runtime/gait_split_manifest.json
```

Expected paper-comparison boundary:

- 22 train subjects
- 8 test subjects
- 426 gait videos total
- 116 gait test videos
- zero subject overlap

A failing check blocks training.

## Accelerator support gate

The upstream harness uses PyTorch's `torch.cuda` APIs. PyTorch's ROCm build intentionally maps those APIs to HIP. The launcher keeps `CUDA_VISIBLE_DEVICES` for NVIDIA and also sets `ROCR_VISIBLE_DEVICES` for AMD; AMD recommends `ROCR_VISIBLE_DEVICES` on Linux. This API compatibility does **not** make every AMD GPU or operating system ROCm-supported.

Before training on AMD, confirm that the exact GPU, operating system, driver, and PyTorch/ROCm versions appear together in AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). As of 2026-09-23, the matrix lists Windows 11 25H2 for Radeon and the [Radeon support page](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/docs/compatibility/compatibility.html) names RX 9000 and select RX 7000 series GPUs. An RX 6700 XT on Windows 10 is outside that listed configuration.

Do not use an unsupported-GPU override as evidence of a reproducible run. Use a GPU/OS/driver combination listed by AMD, or a supported CUDA machine, then verify the active PyTorch backend before starting this launcher:

```bash
python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "No CUDA/HIP GPU is visible to PyTorch"
backend = "ROCm/HIP" if torch.version.hip else "CUDA" if torch.version.cuda else "unknown"
assert backend != "unknown", "PyTorch reports a GPU but not a CUDA or HIP build"
print(f"backend={backend}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"hip={torch.version.hip}")
print(f"cuda={torch.version.cuda}")
PY
```

## Phase 1 — reproduce CoRe without PECoP

1. clone the pinned CoRe revision
2. adapt its dataset loader to PD4T Gait without changing scoring semantics
3. initialize the same domain-general I3D backbone
4. train/evaluate using the original 22/8 split
5. report SRCC and exact runtime configuration

Gate: investigate preprocessing, validation, sampling, and score normalization until the result is explainable before adding PECoP.

## Phase 2 — reproduce PECoP pretraining

Upstream PECoP uses Kinetics-pretrained I3D plus trainable 3D adapters and self-supervised playback/segment prediction.

Paper-faithful settings reported in the WACV paper:

- epochs: 8
- batch size: 16
- optimizer: SGD
- learning rate: 0.001
- clip_len: 32
- crop: 224
- VSPP playback parameter λ: 4
- VSPP segment parameter ζ: 4 or 3 depending on task
- gait frame sampling rate in the public code: 3

### Known upstream code/paper discrepancies

The current upstream repository does not encode all of those values faithfully:

1. the README calls `train.py --lr 0.001`, but `train.py` constructs SGD with a hard-coded `lr=0.01`; the scheduler still reads `args.lr`
2. the current `train.py` default for `max_sr` is 5, while the paper reports λ=4

Therefore record at least two sensitivity conditions:

- **paper-faithful**: optimizer LR 0.001, λ=4
- **current-code**: optimizer LR 0.01, `max_sr=5`

Keep all other settings identical. Never silently patch or collapse this difference.

## Phase 3 — CoRe + PECoP

Load the PECoP-pretrained I3D backbone into the same downstream CoRe harness used in Phase 1. The only intended experimental difference should be the backbone initialization.

## Validation ambiguity

The paper specifies the 22/8 train/test subject split but the public description does not fully specify an internal validation split. Keep the 8 test subjects sealed and compare at least:

- R1: split validation only inside the 22 train subjects
- R2: use all 22 subjects for final training when the upstream implementation supports it

Record model-selection rules for each run.

## Reporting

Append one row per run to `results/reproduction_matrix.csv`.

Minimum fields:
- upstream commit SHAs
- task
- baseline
- PECoP on/off
- split manifest checksum
- validation policy
- seed
- preprocessing/sampling
- optimizer/lr
- epochs
- SRCC
- notes

A reproduction claim requires a complete configuration, not only a score.
