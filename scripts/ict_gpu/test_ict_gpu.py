"""Dependency-free offline tests. Never contact an ICT device or model endpoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c0b_cuda import (
    PreflightError, check_host, launch_args, parse_compute_gpu_uuids,
    parse_gpu_csv, verify_manifest,
)
from c0b_route_smoke import inference_payload, synthetic_png, validate_base_url


class GuardTests(unittest.TestCase):
    def test_parse_gpu_csv(self):
        rows = "0, GPU-111, 14500, 3\n1, GPU-222, 16010, 0"
        self.assertEqual(parse_gpu_csv(rows)["uuid"], "GPU-111")

    def test_reject_missing_or_invalid_gpu(self):
        with self.assertRaises(PreflightError):
            parse_gpu_csv("1, GPU-222, 16010, 0")
        with self.assertRaises(PreflightError):
            parse_gpu_csv("0, GPU-111, N/A, 0")

    def test_busy_compute_gpu_list(self):
        self.assertEqual(parse_compute_gpu_uuids("GPU-111, 321\nGPU-222, 44"),
                         {"GPU-111", "GPU-222"})
        self.assertEqual(parse_compute_gpu_uuids(""), set())
        with self.assertRaises(PreflightError):
            parse_compute_gpu_uuids("unknown")

    def test_host_guard_wrong_user_or_host(self):
        with self.assertRaises(PreflightError):
            check_host("intentionally-wrong-host", "intentionally-wrong-user")

    def test_sha256_pin_and_modified_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            for role in ("server", "model", "mmproj", "adapter"):
                path = root / role
                path.write_bytes(("synthetic-" + role).encode())
                if role == "server":
                    path.chmod(0o700)
                files[role] = {"path": str(path), "sha256":
                               hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": files}))
            self.assertEqual(set(verify_manifest(manifest)), set(files))
            (root / "adapter").write_bytes(b"changed")
            with self.assertRaisesRegex(PreflightError, "ASSET_SHA256_MISMATCH"):
                verify_manifest(manifest)

    def test_launch_args_foreground_local_gpu(self):
        paths = {r: "/tmp/" + r for r in ("server", "model", "mmproj", "adapter")}
        args = launch_args(paths, 18080, 8192)
        self.assertIn("127.0.0.1", args)
        self.assertIn("--n-gpu-layers", args)
        self.assertEqual(args[args.index("--parallel") + 1], "1")
        self.assertNotIn("0.0.0.0", args)

    def test_no_public_model_endpoint(self):
        self.assertEqual(validate_base_url("http://127.0.0.1:18080/v1", False),
                         "http://127.0.0.1:18080/v1")
        for base in ("https://example.com:443/v1", "http://192.168.0.2:18080/v1",
                     "http://100.83.147.56:18080/v1"):
            with self.assertRaises(ValueError):
                validate_base_url(base, False)
        self.assertEqual(validate_base_url("http://100.83.147.56:18080/v1", True),
                         "http://100.83.147.56:18080/v1")

    def test_synthetic_input_no_patient_data(self):
        png = synthetic_png()
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(png.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82"))
        payload = inference_payload("hawkeye-c0b-seed42")
        images = [x for x in payload["messages"][0]["content"] if x["type"] == "image_url"]
        self.assertEqual(len(images), 8)
        self.assertTrue(all(img["image_url"]["url"].startswith("data:image/png;base64,")
                            for img in images))


if __name__ == "__main__":
    unittest.main()
