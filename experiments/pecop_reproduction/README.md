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

## Phase 1 — reproduce CoRe without PECoP

1. clone the pinned CoRe revision
2. adapt its dataset loader to PD4T Gait without changing scoring semantics
3. initialize the same domain-general I3D backbone
4. train/evaluate using the original 22/8 split
5. report SRCC and exact runtime configuration

Gate: investigate preprocessing, validation, sampling, and score normalization until the result is explainable before adding PECoP.

## Phase 2 — reproduce PECoP pretraining

Upstream PECoP uses Kinetics-pretrained I3D plus trainable 3D adapters and self-supervised playback/segment prediction.

Paper/repo defaults to preserve first:

- clip_len: 32
- crop: 224
- max playback rate: 5
- max segments: 4
- gait sampling rate: 3
- batch size in README example: 16

### Known upstream learning-rate discrepancy

The upstream README example calls `train.py --lr 0.001`, while the current upstream `train.py` constructs SGD with a hard-coded `lr=0.01`. The scheduler still reads `args.lr`.

Record both:
- **code-faithful**: optimizer LR 0.01
- **README-intent sensitivity**: optimizer LR 0.001

Never silently patch this difference.

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
