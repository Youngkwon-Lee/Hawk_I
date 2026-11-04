# Dataset EDA — HUBU-FIS vs PD4T finger tapping

Cross-dataset manifest builder and sanitized summary for the finger-tapping corpora.
Moved from `hawkeye-labeling-tool` on 2026-11-04 (source HEAD `d7ff130`): research analysis belongs to this repo, not the standalone labeler.

## Data boundary

**Code only.** Nothing produced by these scripts is committed here.

- Inputs resolve through `scripts/kinelo_data.py` from `KINELO_DATA_ROOT` (the home-desktop data drive).
- `--output-dir` defaults to the data drive. The detailed manifests (`*_manifest_local_only.csv`) carry filenames and subject identifiers and must stay there.
- HUBU-FIS clip-level governance approval is **pending**. Do not publish results derived from it — including the summary JSON and any notebook narrative — until that approval is recorded.

## Run

```bash
KINELO_DATA_ROOT=/mnt/d python3 experiments/dataset_eda/scripts/build_hubu_pd4t_comparison_eda.py
```

Requires `ffprobe` on `PATH`.

## Test

```bash
python3 -m unittest discover -s experiments/dataset_eda/tests
```

These tests are not under `pytest.ini`'s `testpaths` (`backend/tests`), so run them directly.
