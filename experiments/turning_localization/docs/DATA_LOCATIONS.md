# Data and artefact locations (pointer registry)

This file points to where things live. **No data is stored in Git** (PD4T: academic use, no redistribution).
SSOT roles: research decisions/summary → second-brain research index (see below); code and this pointer
file → this repo; meeting status → Notion (HawkEye 회의록); files → D drive / Google Drive (yonsei).
Last verified: 2026-11-12.

## Machines

| name | access | role |
|---|---|---|
| home desktop (DESKTOP-T43SN5M, WSL) | `ssh yk@100.125.26.99` (Tailscale) | data, pose extraction, CPU training/eval |
| ICT (WIN-B8QF2DSGDAG, WSL user windowadmin3) | Desktop Commander (WSL agent, or Windows agent + `wsl -u windowadmin3`) | GPU (2× Quadro RTX 5000 16 GB, shared with other jobs) |
| MacBook (M4 Pro 48 GB) | local | repo checkout, orchestration |

## Datasets (home desktop unless noted)

| what | path | notes |
|---|---|---|
| PD4T videos | `/mnt/d/1_PD4T/PD4T/PD4T/Videos/Gait/<pid>/<visit>.mp4` | academic use only |
| PD4T official split | `/mnt/d/1_PD4T/PD4T/PD4T/Annotations/Gait/{train,test}.csv` | 22 / 8 subjects |
| 20-s gait clips (labelled) | `/mnt/d/clip/gait/<pid>/<visit>_<pid>_clip.mp4` (= `D:\clip\gait`) | 426 clips |
| clinician labels | `/home/yk/gait_export/{train,validation,test}.jsonl` | annotator dongsung; turning/freezing intervals, 8 primitives, own 3.10, note. File keeps the old Hawkeye split; code re-splits by the official CSVs |
| TULIP (CVPR 2025) | `D:\1_TULIP`; gait = `Subject_*/25. Gait/Camera1.mp4` | 12 subjects, 3-clinician UPDRS |
| public FoG turning-in-place | `D:\1_FoG_TurningInPlace` (Videos.zip, IMU.zip, PDFEinfo.csv) | Ribeiro De Souza 2022, figshare 14984667, md5 verified 2026-11-12, not unzipped |
| REMAP (Your Turn source) | `D:\1_REMAP\Turning` | 1,749 trimmed turns, 17-joint 2D/3D skeletons, human labels xlsx |

## Derived artefacts (private, not in Git)

| what | path |
|---|---|
| MediaPipe pose (PD4T, 15 fps) | home `~/private/turnpose/npz/*.npz` |
| BiGRU CV/test results | home `~/private/turnpose/sup/` |
| evaluation reports | home `~/private/turnpose/{eval_report,sanity_20261111,yourturn_baseline,updrs_from_turns,fusion_c0b}.json` |
| VLM predictions (test 116) | home `~/private/turnpose/qwen3vl4b_*_test116.jsonl` (+ `.eval.json`, `.paired.json`) |
| TULIP pose + predictions | home `~/private/turnpose/tulip/` (`npz/`, `tulip_turns.json`, `qa_*.jpg`) |
| repo copy used for eval runs | home `~/private/turnpose/repo_eval/` |
| Qwen3-VL turn work dir | ICT `~/private/hawkeye-c0b-qwen3-20261110/turnvlm/` (`clips/` test, `train_clips/`, `ft_r1/adapter_epoch{1,2}`, predictions, logs) |
| Qwen3-VL C0B retrain (PR #37) | ICT `~/private/hawkeye-c0b-qwen3-20261110/runs/c0b-qwen3-ict-20261111-r002/` |
| base model | ICT `~/private/hawkeye-c0b-qwen3-20261110/models/Qwen3-VL-4B-Instruct` |

## Code and documents

| what | where |
|---|---|
| this repo (private) | `github.com/Youngkwon-Lee/pd4t-turn-localizer` (Mac `~/projects/pd4t-turn-localizer`) |
| HawkEye mirror of the experiment | `Hawk_I` PR #40, `experiments/turning_localization/` |
| C0B retrain pipeline | `Hawk_I` PR #37 |
| study design | `STUDY_DESIGN.md` |
| results summary | `docs/RESULTS.md` |
| 15th meeting slides | Claude artifact "HawkEye 15차 회의 — 회전 primitive 연구" |
| meeting notes | Notion 회의록 › 15차 회의 사전 안건 (status view, not SSOT) |
| Drive copy (share/archive, not SSOT) | yonsei Drive › 10_research/active/HawkEye_PD_VLM/2026-11-12_turn_primitive_study (https://drive.google.com/drive/folders/1iQDtn1w_LS1r57x9zz154f0X7g5Aime4) |
| research index (decisions) | second-brain `candidates/2026-11-12-hawkeye-turn-primitive-research-index.md` (PR #657, pending review) |
