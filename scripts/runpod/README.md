# Runpod Qwen 7B Notes

These helpers are for running `Qwen/Qwen2.5-VL-7B-Instruct` as an OpenAI-compatible backend for Hawkeye teacher prefill.

## Recommended Path

Use a **Runpod Pod** with HTTP port `8000` exposed and a custom image that already contains the Qwen VL server.

Primary files:

- `scripts/runpod/qwen_vl_openai_server.py`
- `scripts/runpod/qwen_vl_fallback_server.py`
- `scripts/runpod/start_qwen_vl_with_fallback.py`
- `scripts/runpod/requirements-qwen-vl-server.txt`
- `scripts/runpod/Dockerfile.qwen-vl-server`

This is more stable than cloning another repo and pip-installing on every pod start.
The VL image now also includes a fallback diagnostics server so `/health` still returns useful startup logs if the main VL app dies before binding.

## Base Image

Earlier Runpod-specific PyTorch base images were useful for debugging, but the current Qwen path now follows the proven GPU smoke-test pattern:

```text
pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
```

This is now the default base for:

- `scripts/runpod/Dockerfile.qwen-vl-server`
- `scripts/runpod/Dockerfile.qwen-text-smoke`

Reason:

- `ghcr.io/<owner>/hawkeye-gpu-hello-health:latest` proved `GPU base + plain CMD + Runpod proxy` works
- `dockerStartCmd` and inline-command experiments remained unreliable
- the current recommendation is to keep Qwen images close to the working smoke-image pattern

## Debugging Path

Before debugging Qwen itself, validate that:

- GHCR image pull works
- Runpod pod startup works
- HTTP port `8000` is correctly attached

Files for this smoke test:

- `scripts/runpod/hello_health_server.py`
- `scripts/runpod/requirements-hello-health.txt`
- `scripts/runpod/Dockerfile.hello-health`
- `.github/workflows/build-hello-health-image.yml`

Files for GPU-base smoke isolation:

- `scripts/runpod/Dockerfile.gpu-hello-health`
- `.github/workflows/build-gpu-hello-health-image.yml`

Expected image:

```text
ghcr.io/<owner>/hawkeye-hello-health:latest
```

If this image responds on `/health`, then GHCR + Runpod HTTP attach is working and the remaining issues are specific to the Qwen server path.

If the GPU-base hello image also responds on `/health`, then the remaining issues are specific to the Qwen runtime or startup logic, not the GPU base image itself.

## Why Pod Instead of Serverless First

- Hawkeye already has a custom VLM OpenAI-compatible server path.
- The current prefill script sends sampled frames, not raw uploaded video, so a simple pod-based HTTP service is enough.
- It avoids extra serverless worker packaging while you are still validating silver-train generation.

## GPU Recommendation

For a 7B model, start with **A10G or better**.

## Pod Setup Checklist

1. Create a Runpod Pod.
2. Choose a GPU with enough VRAM for Qwen2.5-VL-7B-Instruct.
3. Build and push the custom image from `Dockerfile.qwen-vl-server`.
4. Create a Pod from that image.
5. Expose HTTP port `8000`.
6. Let the container start normally with its default `CMD`.

## Base URL

For a pod HTTP proxy, use:

```text
https://YOUR_POD_ID-8000.proxy.runpod.net/v1
```

That becomes `OPEN_MODEL_BASE_URL`.

## Health Check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\runpod\check_hawkeye_qwen_server.ps1
```

## Silver Prefill Smoke Test

```powershell
python scripts/data/generate_motorexam_teacher_prefill.py `
  --candidates data/processed/motorexam_vqa/silver_train_candidates_v0_1.csv `
  --output data/processed/motorexam_vqa/prefill/silver_train_prefill_v0_1.jsonl `
  --mode runpod `
  --base-url https://YOUR_POD_ID-8000.proxy.runpod.net/v1 `
  --api-key EMPTY `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --limit 10 `
  --question-limit 2
```

## Gold Reviewer-Assist Smoke Test

```powershell
python scripts/data/generate_motorexam_teacher_prefill.py `
  --candidates data/processed/motorexam_vqa/gold_benchmark_candidates_v0_1.csv `
  --output data/processed/motorexam_vqa/prefill/gold_benchmark_prefill_v0_1.jsonl `
  --mode runpod `
  --base-url https://YOUR_POD_ID-8000.proxy.runpod.net/v1 `
  --api-key EMPTY `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --limit 10 `
  --question-limit 2
```

## Next Step After Bring-Up

Once the endpoint responds, the next thing to do is:

1. run a `limit 5-10` silver prefill test
2. inspect output JSONL
3. open the result in Kinelo Lab for review
