# MotorExam-VQA Runpod Qwen2.5-VL-7B Setup v0.1

**Updated**: 2026-03-28  
**Author**: Codex (GPT-5)  
**Status**: Draft v0.1

---

## 1. Purpose

This document explains how to use **Qwen/Qwen2.5-VL-7B-Instruct** as the first practical open-model backend for:

- `silver train` teacher prefill
- optional `gold` reviewer assistance

in the MotorExam-VQA pipeline.

---

## 2. Recommendation

For the current Hawkeye stage, the recommended strategy is:

- `gold benchmark prefill`: Gemini when available
- `silver train scale-up`: Runpod / OpenAI-compatible Qwen2.5-VL-7B-Instruct

If Gemini is blocked by budget, Qwen 7B can be used as the practical first operational model.

---

## 3. Why 7B

### Recommended First Open Model

- `Qwen/Qwen2.5-VL-7B-Instruct`

### Why Not 3B

- lower reasoning quality
- weaker evidence grounding
- more unstable rationale draft quality

### Why Not 72B First

- much heavier serving cost
- more operational friction
- unnecessary for first silver-train iteration

The 7B model is the best first compromise between:

- quality
- speed
- VRAM cost
- deployment simplicity

---

## 4. Important Practical Note

Even if the model family supports video understanding, your actual serving stack may still be easiest to operate using:

- sampled frames
- OpenAI-compatible image inputs

That is exactly how the current Hawkeye prefill script is prepared to run in `runpod` mode.

---

## 5. Hawkeye-Native Runtime Assets

Runpod-related assets now live directly in the Hawkeye repo:

- `scripts/runpod/qwen_vl_openai_server.py`
- `scripts/runpod/start_hawkeye_qwen_server.ps1`
- `scripts/runpod/start_hawkeye_qwen_server.sh`
- `scripts/runpod/check_hawkeye_qwen_server.ps1`
- `scripts/runpod/Dockerfile.qwen-vl-server`
- `scripts/runpod/.env.runpod.example`

### Recommended Runpod Base Image

For current Qwen bring-up, Hawkeye uses the same GPU base family that worked in the `gpu-hello-health` smoke image:

```text
pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
```

Reason:

- `GPU base + plain CMD + Runpod proxy` was verified end-to-end with the GPU hello-health smoke image
- earlier `runpod/pytorch` plus startup-command variants stayed unreliable
- the current recommendation is to keep Qwen images close to the proven smoke-image pattern

---

## 6. Local / Runpod Serving Pattern

### Example vLLM Launch Pattern

Reference:

```powershell
python -m vllm.entrypoints.openai.api_server `
  --host 127.0.0.1 `
  --port 8000 `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --trust-remote-code `
  --max-model-len 8192 `
  --limit-mm-per-prompt image=1
```

This is suitable when:

- you are serving an OpenAI-compatible endpoint
- the prefill script sends frame samples instead of raw video upload

---

## 7. Minimum Environment Variables

For Hawkeye teacher prefill with Runpod/open-model mode, the key settings are:

```env
OPEN_MODEL_BASE_URL=http://YOUR_ENDPOINT/v1
OPEN_MODEL_API_KEY=YOUR_KEY_OR_EMPTY
OPEN_MODEL_GENERATE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

If the endpoint is local:

```env
OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1
OPEN_MODEL_API_KEY=EMPTY
OPEN_MODEL_GENERATE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 8. Hawkeye Teacher Prefill Command

### Silver Train Small Test

```powershell
python scripts/data/generate_motorexam_teacher_prefill.py `
  --candidates data/processed/motorexam_vqa/silver_train_candidates_v0_1.csv `
  --output data/processed/motorexam_vqa/prefill/silver_train_prefill_v0_1.jsonl `
  --mode runpod `
  --base-url http://YOUR_ENDPOINT/v1 `
  --api-key YOUR_KEY `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --limit 20 `
  --question-limit 2
```

### Gold Reviewer Assistance Small Test

```powershell
python scripts/data/generate_motorexam_teacher_prefill.py `
  --candidates data/processed/motorexam_vqa/gold_benchmark_candidates_v0_1.csv `
  --output data/processed/motorexam_vqa/prefill/gold_benchmark_prefill_v0_1.jsonl `
  --mode runpod `
  --base-url http://YOUR_ENDPOINT/v1 `
  --api-key YOUR_KEY `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --limit 10 `
  --question-limit 2
```

---

## 9. Recommended Rollout Order

### Step 1

Run a very small test:

- `limit 5-10`

Check:

- output JSON parses
- evidence spans look sane
- cue tags are not drifting badly

### Step 2

Run a medium silver batch:

- `limit 50-100`

Check:

- reviewer burden
- average correction rate

### Step 3

Scale silver train:

- `100+`
- eventually the full candidate pool

---

## 10. Expected Quality Level

### Good Use Cases for Qwen 7B

- silver train prefill
- answer proposal
- initial span proposal
- rationale draft

### Use With More Caution

- difficult benchmark edge cases
- borderline score decisions
- ambiguous evidence localization

That is why Qwen 7B is best used first for:

- `silver`

and only secondarily for:

- `gold reviewer assistance`

---

## 11. Current Blocker

At the moment, Hawkeye is ready for Runpod/open-model generation in code, but a live OpenAI-compatible endpoint must exist.

Current status observed:

- env pattern exists
- script support exists
- endpoint was not running
- no active Runpod endpoint or pod was found

So the remaining operational requirement is simply:

- launch or connect a Qwen 7B endpoint

When debugging image/runtime issues, prefer staying close to the proven GPU smoke-image pattern above before reintroducing startup-command complexity.

---

## 12. Final Recommendation

Yes, **Qwen/Qwen2.5-VL-7B-Instruct** is the correct first model to use for MotorExam-VQA silver-train generation.

Use it as:

- the first open-model teacher backend
- frame-sampled OpenAI-compatible inference
- silver-train prefill generator

This is the most practical next step after the current Hawkeye pipeline work.
