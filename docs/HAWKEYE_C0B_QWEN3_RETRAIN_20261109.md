# Hawk_I C0B Qwen3-VL gait retraining — November 2026

Status: **private preparation implementation; no real-data training run, no verified model.**

## Scope

This implements a **new, separately versioned research training path**, not retroactive verification of the archived August 2026 model.

- Base architecture: Qwen/Qwen3-VL-4B-Instruct, **not** the repository's older Qwen2.5-VL training script.
- Research condition: C0B (video + the existing gait scoring anchor/glossary; **no** clinician or kinematics features).
- Prompt contract: imports backend/services/vlm_training_contract.py; does not put GT in the user prompt.
- Training engine: official ms-swift Qwen3-VL LoRA SFT; **local CUDA only**, no hosted patient-video service.
- Outputs: private train.jsonl, validation.jsonl, private_media_binding.jsonl, run.json with hashes, counts and **explicitly unverified** state.
- A held-out test portion is required to check declared split disjointness, but test videos/labels are **never** provided to the trainer or to validation.

This is **not** an equivalent replication of 2026-08-17 C0B: the new pilot setting samples at most 16 video frames (via ms-swift), while the archived experiment describes 5 FPS / up to 100 frames / width 512 px. Treat any result as a separate experiment. Do not mix it with the archived report or claim benchmark improvement.

## Owner and data-use boundary

Before preparing *real* examples, the study custodian must provide a frozen subject-level manifest and confirm applicable research use. No study media is uploaded by this script, but local hashing and training are still data processing. No model or patient video is copied into Git or a public PR.

**Required manifest JSONL schema** (these are fictional, synthetic examples):

    {"clip_id":"synthetic-train-1","subject_id":"synthetic-person-1","split":"train","dataset":"PD4T","task":"gait","score":1,"video_path":"synthetic-train-1.mp4"}
    {"clip_id":"synthetic-val-1","subject_id":"synthetic-person-2","split":"validation","dataset":"PD4T","task":"gait","score":2,"video_path":"synthetic-val-1.mp4"}
    {"clip_id":"synthetic-test-1","subject_id":"synthetic-person-3","split":"test","dataset":"PD4T","task":"gait","score":0,"video_path":"synthetic-test-1.mp4"}

Supply corresponding private MP4 files under the declared media root. The script does not infer the subject from a filename; it rejects duplicate clip IDs or paths, out-of-root paths, missing splits, non-integer grades, and subject leakage across train/validation/test. Grade range 0–4.

**No automatic re-split.** The data owner must supply the existing approved split roster and its origin. A matching source manifest SHA-256 proves which local file was used, not that the roster is independently correct or that a human scored the clips reliably.

## Prepare offline in a private workspace

    # On the authorized research machine; use absolute paths outside the Git checkout:
    shasum -a 256 /private/study/pd4t_gait_frozen.jsonl

    python scripts/vlm/c0b_retrain.py prepare \
      --manifest /private/study/pd4t_gait_frozen.jsonl \
      --media-root /private/study/videos \
      --run-root /private/hawkeye-runs \
      --run-id c0b-qwen3-20261109-r001 \
      --expected-manifest-sha256 <the-exact-64-character-SHA256>

Preparation reads MP4 bytes locally to freeze per-clip content hashes and writes restricted private files. It reports only aggregate counts. A missing clip or unresolved split is a hard error; do not insert replacement placeholder media.

## Explicit training execution (NOT performed as part of this implementation)

Use preinstalled and study-approved local Hugging Face Qwen3-VL-4B weights and the ms-swift>=4 training engine. A GGUF converted adapter is **inference-only**. The script fingerprints local base weights/config before execution, disables external model downloads and tracking, and checks for GPU contention.

An authorized data steward must supply a **real, private approval JSON** with these keys:

    {"dataset":"PD4T","approved":true,"purpose":"local_research_retraining","issued_by":"<real authorized steward>","approval_reference":"<existing IRB/DUA reference>","approved_at":"<actual date>"}

This is a schema example, **not** an approval. The script checks required fields and records a SHA-256. It does not cryptographically verify signer identity or independently determine patient-consent/IRB scope; a responsible human must do so before actual data use.

    # Read-only prepared-data check; does not start training:
    python scripts/vlm/c0b_retrain.py train \
      --run-dir /private/hawkeye-runs/c0b-qwen3-20261109-r001 \
      --base-model-dir /private/models/Qwen3-VL-4B-Instruct \
      --approval-json /private/study/data-use-approval.json \
      --gpu-index 0

    # Only after real data-use authority AND assigned GPU owner approval:
    python scripts/vlm/c0b_retrain.py train \
      --run-dir /private/hawkeye-runs/c0b-qwen3-20261109-r001 \
      --base-model-dir /private/models/Qwen3-VL-4B-Instruct \
      --approval-json /private/study/data-use-approval.json \
      --gpu-index 0 \
      --compute-approved \
      --execute

The model engine invokes swift sft on a fixed frozen train and validation dataset, explicitly disables further auto-splitting, and uses LoRA rank 8 / alpha 32, frozen vision tower and aligner, bfloat16, one epoch, batch 1, gradient accumulation 2, a 16-frame/128-video-token pilot budget, and max sequence length 4096. This budget differs from the archived high-token study. Do not claim the two settings are comparable without controlled study.

The prepared training files include private paths; only local, restricted output folders and logs may contain them. The script does not touch the running HawkEye server, Vercel, a clinical database, or patient timelines.

## Evidence and gates

1. Preparation: source manifest SHA-256, private per-clip media hashes, declared patient-disjoint splits, aggregate record/subject counts; held-out provenance remains unverified.
2. Training: data-steward authorization hash/reference, base-model weight/config hashes, exact training run, process exit status, adapter weights and SHA-256, private logs.
3. Independent evaluation: actual frozen held-out participant split, media-label custody, expert reference reliability, per-grade performance, QWK/MAE, uncertainty, error analysis. Never tune repeatedly on test.
4. Promotion: data-use and clinical authority are separate. No model clinical use without task-specific validity, human oversight and permission.

The previously recovered August 2026 LoRA artifact stays untouched. The October 2026 GGUF/Vulkan files prove runtime setup, not October/November training. Existing Draft PR #26 covers ICT **inference** migration, not this training job.

Official ms-swift Qwen3-VL reference:
https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md

**Verification scope:** new unit and CLI checks use only synthetic subjects/MP4 placeholders. No patient media were opened; no training GPU was reserved, no actual model trained, and no production services changed.
