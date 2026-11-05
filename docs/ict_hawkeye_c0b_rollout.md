# ICT CUDA C0B inference: bounded rollout and rollback

Status (2026-11-05): **offline preparation only**. This is not an ICT installation, live-model validation, production deployment, or clinical data transfer. Review before merge.

## Goal and ownership

Keep ParkiCheck browser MediaPipe tests, Vercel, and the Hawk_I API unchanged. Move only costly C0B VLM inference to an isolated ICT GPU0 process after validation. Protect ICT GPU1 ACT, existing CI/education/experiments, remote-desktop operator tabs, installed drivers, packages, services, cache and device ownership.

Existing topology:
- ParkiCheck browser -> Hawk_I API -> fine-tuned VLM.
- Hawk_I reads HAWKEYE_VLM_BASE_URL / HAWKEYE_VLM_MODEL. On inference failure it returns the existing rule-based analysis.
- Home Windows Vulkan llama.cpp uses Qwen3-VL-4B Q4 GGUF, mmproj Q8 GGUF and hawkeye-c0b-seed42 F16 LoRA. A **separate CUDA llama.cpp binary** is necessary for ICT Turing. CPU-only fallback is not an acceptable GPU verification.
- /api/vlm/finetuned/status measures configured values, not model health. Use /api/pipeline/status for a live probe and a synthetic model inference.

At 2026-11-05 22:26 KST the home endpoint timed out, while the Hawk_I API returned healthy. Public /api/pipeline/status reported model.reachable=false, model_present=false, inference.ready=false. That snapshot is not an ICT GPU measurement.

## Non-negotiable gates

1. Acquire a fresh **authorized independent ICT Linux shell**, with the precise host and OS user confirmed. Do not hijack another RDP operator tab or another process, GPU reservation or CI job. The Mac and home Windows devices are not the ICT NVIDIA host.
2. Verify the exact CUDA llama.cpp binary ABI on the ICT Turing GPU, model asset license/transfer rights, and availability of all four source-verified files. Neither script installs packages, transfers/downloads weights nor changes system drivers.
3. Create a **private, operator-owned** manifest with absolute paths and expected SHA-256 fingerprints from the approved source. Do not commit actual paths, credentials, model weights or large files. Default preflight validates every hash and refuses symlinks.
4. Coordinate with ICT GPU0 owner. The script checks GPU0 >=12000 MiB free, <=5% utilization and no active GPU0 compute process immediately before serving; it **fails closed** if query is unavailable. This does not reserve GPU. GPU1 is not selected. CUDA_VISIBLE_DEVICES is pinned to the measured GPU0 UUID, not enumeration order.
5. Bind the foreground server to 127.0.0.1:18080 only, parallel=1, context=8192. Do not start systemd, change firewall, expose a public tunnel, edit other worktrees or take control of a shared RDP session.
6. Conduct a model-list and explicit 8-image **synthetic** inference smoke. Verify server GPU0 memory and actual inference, not only advertised model list. Synthetic results do not prove Parkinson assessment accuracy, model training/data provenance, clinical readiness or sustained latency.
7. Only create a private cross-host path once operator access, Tailscale/SSH policy and health-data processing permission are confirmed. No real patient video transfer without consent and approved purpose, minimization and retention controls.

## Operator-owned manifest (example, not real hashes)

    {
      "files": {
        "server":  {"path": "/absolute/path/to/cuda-llama-server", "sha256": "<64 lowercase hex>"},
        "model":   {"path": "/absolute/path/to/Qwen3VL-4B-Instruct-Q4_K_M.gguf", "sha256": "<64 lowercase hex>"},
        "mmproj":  {"path": "/absolute/path/to/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", "sha256": "<64 lowercase hex>"},
        "adapter": {"path": "/absolute/path/to/hawkeye-c0b-seed42-f16.gguf", "sha256": "<64 lowercase hex>"}
      }
    }

## Reproducible ICT preflight and isolated inference

Once approved files and CUDA server are present in a new isolated location and the shared host operator has confirmed ownership, run in the **authorized ICT shell**:

    python3 scripts/ict_gpu/c0b_cuda.py \
      --expected-host "$ICT_EXPECTED_HOST" --expected-user "$ICT_EXPECTED_USER" \
      --manifest /absolute/private/manifest.json

Preflight runs without starting a service. Then, only after coordination, run the same command with explicit --serve to start the foreground model process:

    python3 scripts/ict_gpu/c0b_cuda.py \
      --expected-host "$ICT_EXPECTED_HOST" --expected-user "$ICT_EXPECTED_USER" \
      --manifest /absolute/private/manifest.json --serve

In a second authorized terminal on the same host:

    python3 scripts/ict_gpu/c0b_route_smoke.py \
      --base-url http://127.0.0.1:18080/v1 --model hawkeye-c0b-seed42
    python3 scripts/ict_gpu/c0b_route_smoke.py \
      --base-url http://127.0.0.1:18080/v1 --model hawkeye-c0b-seed42 --infer

The second call uses 8 generated solid-gray PNG images, never patient footage. It verifies transport/nonempty response, not clinical output. Check nvidia-smi GPU0 memory and process identity while it runs. If CPU-only or wrong physical GPU, fail acceptance.

## Promotion to the existing home Hawk_I backend

Keep the ParkiCheck/Vercel path unchanged. Only after the authorized ICT model smoke and secure private connectivity are verified from the **home WSL service user**:

1. Capture existing HAWKEYE_VLM_BASE_URL, model, frames, timeout, current process state, and a private rollback receipt. Do not log secrets.
2. Run synthetic c0b_route_smoke.py FROM HOME WSL against the private approved tailnet IP, with --allow-tailnet. Verify the exact endpoint ACL and SSH/tunnel permissions separately; never expose a public port.
3. With service-owner approval, apply the new VLM URL only in an owned home systemd drop-in, preserve model=hawkeye-c0b-seed42, C0B, max_frames=8 and timeout, then daemon-reload/restart the owned Hawkeye backend using the normal service-change contract.
4. Check /api/health and /api/pipeline/status with model.reachable/model_present; then a synthetic **Hawkeye end-to-end** analysis proves result source finetuned_vlm. Validate clinical task behavior separately. Measure latency/VRAM before claiming improvement.
5. Keep the existing rule-based fallback when ICT fails. Do not let retry/timeout storms degrade the interactive ParkiCheck app.

## Rollback / completion boundary

If any gate fails before cutover, do not change production. After a later cutover, revert only the owned VLM URL drop-in to the saved prior value, daemon-reload/restart the owned backend during its permitted window, and confirm /api/health plus /api/pipeline/status. The prior VLM itself may also be offline; the existing rule scorer remains the fallback. Never terminate another operator's GPU process.

This PR covers **code + offline tests only**. It does not imply ICT server access, file placement, CUDA model start, actual GPU 0 process, live inference, secure tunnel, production routing, end-to-end validation, or main merge.

Offline checks:

    python3 -m unittest discover -s scripts/ict_gpu -p "test_ict_gpu.py" -v
