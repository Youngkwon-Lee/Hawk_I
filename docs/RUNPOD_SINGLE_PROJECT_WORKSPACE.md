# RunPod Single Project Workspace

This note describes the recommended RunPod layout for **Hawkeye only**.

## Recommended layout

```bash
/workspace
  /hawkeye
  /data
    /PD4T
      /PD4T
        /PD4T
  /results
    /hawkeye
  /logs
    /hawkeye
  /cache
    /huggingface
```

## Why this layout works

- Keep Hawkeye as the only project repo on the Pod
- Keep dataset and cache outside the repo
- Avoid cross-project contamination
- Make Pod recreation simpler

## Quick setup

```bash
mkdir -p /workspace/data /workspace/results/hawkeye
mkdir -p /workspace/logs/hawkeye /workspace/cache/huggingface
```

## Base image note

For current Hawkeye Qwen images, prefer the GPU base that already worked with the `gpu-hello-health` smoke image:

```text
pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
```

This keeps the runtime closer to the proven `GHCR image -> Runpod Pod -> HTTP health 200` path.

## Helper scripts

### Start Hawkeye Qwen server

```bash
cd /workspace/hawkeye
bash scripts/runpod/start_hawkeye_qwen_server.sh
```

### Extract Hawkeye finger tapping features

```bash
cd /workspace/hawkeye
bash scripts/runpod/run_finger_feature_extract.sh
```

## Suggested habit

- Short pause: stop the Pod
- Longer pause: keep data outside the repo and recreate later
- Repeated workflow: keep the same Hawkeye-only folder layout every time
