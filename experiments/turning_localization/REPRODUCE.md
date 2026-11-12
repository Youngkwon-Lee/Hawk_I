# Reproducing the results

Every number in `docs/RESULTS.md` comes from the scripts below, run in this order. Inputs are pinned by
`repro/input_manifest.json` (sha256 of labels, split CSVs, the 426 clips, pose files, checkpoints and result
files) — rerun `training/make_manifest.py` and compare before trusting a rerun. Data locations:
`docs/DATA_LOCATIONS.md`. Environments: `repro/requirements-home-desktop.txt` (CPU) and
`repro/requirements-ict-gpu.txt` (GPU). PD4T data are academic-use only and never committed.

Paths come from environment variables read in `training/config.py` (`PD4T_TURN_ROOT`, `PD4T_CLIP_ROOT`,
`PD4T_TURN_LABEL_DIR`, `PD4T_OFFICIAL_ANN`); the Qwen scripts read `QWEN3VL_MODEL`.

## A. CPU (home desktop), from `training/`

| step | command | output | seeds / determinism |
|---|---|---|---|
| 1 pose | `python extract_pose.py` | `npz/*.npz` | MediaPipe VIDEO mode, deterministic on the same build |
| 2 rules | `python rule_height.py`, `python rule_path.py` | rule predictions | deterministic |
| 3 BiGRU CV + test | `python turn_sup.py` | `sup/results.json` | CV seeds 0–4, test seeds 100–102 |
| 4 final checkpoints | `python train_final.py` | `models/turn_bigru_seed{100,101,102}.pt` | seeds 100–102; reproduced test predictions exactly (116/116) |
| 5 evaluation report | `python eval_report.py` | `eval_report.json` | bootstrap rng 0; shuffled-label control seed 7/100 |
| 6 Your Turn baseline | `python yourturn_baseline.py` | `yourturn_baseline.json` | deterministic grid on train |
| 7 duration agreement | `python duration_agreement.py <vlm.jsonl>` | stdout | deterministic |
| 8 UPDRS from turns | `python updrs_from_turns.py` | `updrs_from_turns.json` | logistic regression, C by grouped CV |
| 9 primitives oracle | `python updrs_from_primitives.py` | stdout | deterministic |
| 10 primitive bottleneck | `python primitive_bottleneck.py` | `primitive_bottleneck.json` | OOF turn models seeds 200–204 |
| 11 fusion with C0B | `python fusion_c0b.py c0b_preds.json` | `fusion_c0b.json` | fixed 50:50 rule |
| 12 TULIP external | `python tulip_external.py extract|apply|updrs` | `tulip/` | frozen checkpoints |
| 12b TULIP bottleneck | `python tulip_bottleneck.py` | `tulip/tulip_bottleneck.json` | rebuilds 2b (OOF seeds 200–204, asserts 0.743 in the log) |
| 12c RGB+pose fusion eval | ICT: `python dump_vlm_primitives.py <turnvlm> <out>` → copy `prim_compact.txt` (check sha256) → `python primitive_fusion_eval.py` | `primitive_fusion_eval.json` | needs `primitive_bottleneck_cache.npz` from step 10 |
| 13 manifest | `python make_manifest.py` | `repro/input_manifest.json` | — |

## B. GPU (ICT), Qwen3-VL-4B

| step | command | notes |
|---|---|---|
| zero-shot turn grounding | `python qwen_turn_zeroshot.py --clips clips --manifest test_manifest_vlm.json --prompt v1|v2 --out …` | greedy decoding; v1 fixed before test scoring, v2 post-hoc |
| turn fine-tune | `bash ft_loop.sh train_clips train_manifest_vlm.json ft_r1 1` | 2 epochs, seed 0, order seed 1000+epoch, resumable |
| turn fine-tuned predict | `python qwen_turn_zeroshot.py … --prompt v2 --adapter ft_r1/adapter_epoch2` | |
| primitive early fusion | `python export_primitive_manifest.py` (home) → `python qwen_primitive_finetune.py train|predict …` | final model + 2-fold OOF; see the script header |
| score VLM predictions | `python eval_vlm.py <pred.jsonl>`, `python compare_paired.py <pred.jsonl>` (home) | same evaluator as BiGRU |

GPU runs are not bit-exact across restarts (resume points, OOM skips on a shared GPU). The run logs
(`train_log.jsonl`: steps, OOM skips, resume events) are kept next to the adapters; report them with results.

## C. What is pinned and where

- clinician labels, official split, clips, pose: hashes in `repro/input_manifest.json`
- turn model weights: `models/*.pt` (hashes in the manifest)
- Qwen adapters and predictions: ICT work dir (`docs/DATA_LOCATIONS.md`), hashes added to the manifest once copied
