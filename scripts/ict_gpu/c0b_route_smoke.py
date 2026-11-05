#!/usr/bin/env python3
"""Read-only model listing, with an optional synthetic (non-patient) C0B transport smoke.

Only loopback or explicitly enabled Tailscale IP targets are allowed. Never
submit real recordings to this script. No credentials, DB or uploads are read.
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import struct
import time
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener
import zlib


def validate_base_url(base_url: str, allow_tailnet: bool) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https")
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"
            or not parsed.hostname or parsed.port is None):
        raise ValueError("INVALID_BASE_URL")
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    is_tailnet = False
    try:
        address = ipaddress.ip_address(hostname)
        is_loopback = is_loopback or address.is_loopback
        is_tailnet = address in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        pass
    if not is_loopback and not (allow_tailnet and is_tailnet):
        raise ValueError("ENDPOINT_MUST_BE_LOOPBACK_OR_EXPLICIT_TAILNET")
    return base_url.rstrip("/")


def synthetic_png(size: int = 64) -> bytes:
    """Create a solid gray RGB PNG without Pillow or external files."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))
    pixels = b"".join(b"\x00" + b"\x80\x80\x80" * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels))
            + chunk(b"IEND", b""))


def request_json(opener, url: str, body: dict | None, timeout: float) -> dict:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=payload, headers={"Content-Type": "application/json"},
                  method="POST" if body is not None else "GET")
    with opener.open(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("NON_200_RESPONSE")
        return json.load(response)


def inference_payload(model: str) -> dict:
    frame = "data:image/png;base64," + base64.b64encode(synthetic_png()).decode("ascii")
    content = [{"type": "text", "text":
                "Transport test only: eight identical synthetic gray frames, not a "
                "patient video. Reply with a single nonempty answer line."}]
    content.extend({"type": "image_url", "image_url": {"url": frame}} for _ in range(8))
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 100, "stream": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="hawkeye-c0b-seed42")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--allow-tailnet", action="store_true",
                        help="allow IPs inside 100.64.0.0/10 (requires private ACL)")
    parser.add_argument("--infer", action="store_true",
                        help="explicitly run one synthetic 8-image model inference")
    args = parser.parse_args()
    try:
        base_url = validate_base_url(args.base_url, args.allow_tailnet)
        if not 1 <= args.timeout <= 300:
            raise ValueError("INVALID_TIMEOUT")
        opener = build_opener(ProxyHandler({}))
        start = time.monotonic()
        models = request_json(opener, base_url + "/models", None, min(args.timeout, 15))
        model_ids = [row.get("id") for row in models.get("data", [])
                     if isinstance(row, dict)]
        if args.model not in model_ids:
            raise RuntimeError("EXPECTED_MODEL_NOT_ADVERTISED")
        report = {"status": "model_list_pass", "model": args.model,
                  "elapsed_ms": round((time.monotonic() - start) * 1000)}
        if args.infer:
            start = time.monotonic()
            reply = request_json(opener, base_url + "/chat/completions",
                                 inference_payload(args.model), args.timeout)
            choices = reply.get("choices", [])
            content = choices[0].get("message", {}).get("content") if choices else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("SYNTHETIC_REPLY_EMPTY")
            report.update(status="synthetic_inference_pass",
                          inference_elapsed_ms=round((time.monotonic() - start) * 1000),
                          reply_nonempty=True)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        # Do not expose error response bodies, remote content, API keys or frames.
        print(json.dumps({"status": "blocked", "reason": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
