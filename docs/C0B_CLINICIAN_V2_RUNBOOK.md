# C0B clinician-v2 retraining runbook

This run retrains the autonomous gait-video condition on the current clinician
labels. It does not overwrite or claim equivalence to the original
`hawkeye-c0b-seed42`, whose target was the original PD4T score.

## Immutable inputs

- handoff: `pd4t-gait-956e5b2add51`
- dataset SHA-256: `956e5b2add51eadbc3ccc5f437c65beeee30b71be8af93dd094e5c6c37d23f53`
- rows: train 284, validation 73, test 69
- split rule: patient-disjoint; overlap must be zero
- target: clinician `updrs_3_10`, integer 0–4
- prompt: `SYSTEM → ANCHOR → GLOSSARY → QUESTION → VIDEO`
- base revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- video preprocessing: 5 fps, width 512, processor resampling disabled,
  per-frame pixel cap enabled
- test: locked until validation-based model selection is frozen

The stage and run manifests bind the dataset SHA, prompt SHA, base-model
revision, code revision, package versions, seed, sampling settings, LoRA
configuration, adapter SHA, and evaluation outputs.

## 1. Stage on the data host

Run where the exported media paths are accessible. Staging copies only the
video, split, and score. It replaces clip/patient identifiers and original
filenames with opaque IDs bound to the dataset SHA.

```bash
python backend/scripts/stage_c0b_training.py \
  --export-dir ~/gait_export \
  --output-dir ~/gait_training_stages/956e5b2add51 \
  --expected-dataset-sha256 956e5b2add51eadbc3ccc5f437c65beeee30b71be8af93dd094e5c6c37d23f53
```

Do not stage into a reused, unverified directory. The command is idempotent only
when the existing stage manifest and every split/media file still verify.

## 2. Validate on the GPU host

Transfer the staged directory through the approved research-data path, then run:

```bash
python scripts/vlm/train_qwen3_c0b_clinician.py validate \
  --data-dir /workspace/hawkeye-c0b-data \
  --expected-dataset-sha256 956e5b2add51eadbc3ccc5f437c65beeee30b71be8af93dd094e5c6c37d23f53
```

The validator refuses changed split files, missing videos, changed prompt text,
wrong labels, duplicate opaque IDs, or absolute/path-traversal media paths.

## 3. Train and evaluate validation

The reference environment is in `scripts/runpod/Dockerfile.qwen3-c0b-train`.
Set the dataset SHA and run the wrapper:

```bash
export HAWKEYE_DATASET_SHA256=956e5b2add51eadbc3ccc5f437c65beeee30b71be8af93dd094e5c6c37d23f53
bash scripts/runpod/run_qwen3_c0b_clinician.sh
```

Outputs include:

- `training-run-manifest.json`
- `adapter_model.safetensors` and `adapter_config.json`
- `predictions-validation.jsonl`
- `metrics-validation.json`

The run stays a `candidate` and is not automatically bound to inference.

## 4. Freeze selection, then open test once

After the candidate and selection rule are frozen, evaluate the held-out split:

```bash
python scripts/vlm/train_qwen3_c0b_clinician.py evaluate \
  --data-dir /workspace/hawkeye-c0b-data \
  --expected-dataset-sha256 956e5b2add51eadbc3ccc5f437c65beeee30b71be8af93dd094e5c6c37d23f53 \
  --run-dir /workspace/runs/hawkeye-c0b-clinician-v2-seed42 \
  --split test \
  --unlock-test
```

Report QWK, MAE, exact agreement, within-1 agreement, balanced accuracy,
GT2+ sensitivity, parse failures, class distributions, confusion matrix, and
the majority-class baseline. Test contains no grade-3 labels, so it cannot
measure grade-3 recall.

## 5. Promotion

Promotion is a separate reviewed operation. Only after validation and test
artifacts are accepted should the adapter be converted for the serving runtime,
smoke-tested against non-patient media, and bound to the dataset SHA in the
pipeline status. Keep the old serving model available for rollback.
