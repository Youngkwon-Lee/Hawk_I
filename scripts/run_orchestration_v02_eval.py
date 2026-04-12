#!/usr/bin/env python3
"""
20-sample evaluator for Hawkeye orchestration v0.2

Compares:
  - baseline (stage1 only)
  - orchestrator Set A
  - orchestrator Set B

Usage:
  python scripts/run_orchestration_v02_eval.py \
    --tracker rationale_results/hawkeye_testset_tracker.csv \
    --hardcase docs/HARDCASE_REGISTRY_V0_2.json \
    --out rationale_results/orch_v02_eval

Requires:
  - GEMINI_API_KEY env var (for live model calls)
  - google-genai package
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ThresholdSet:
    name: str
    sampling_mode_ratio_threshold: float
    disagreement_guardrail_gap: int


SET_A = ThresholdSet("set_a", 0.67, 2)
SET_B = ThresholdSet("set_b", 0.75, 1)


def load_tracker(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_sample_map(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Pick first row per (sample_id, task) for metadata (video_path, gt)."""
    out = {}
    for r in rows:
        k = (r.get("sample_id", ""), r.get("task", ""))
        if k not in out:
            out[k] = {
                "sample_id": r.get("sample_id", ""),
                "task": r.get("task", ""),
                "video_path": r.get("video_path", ""),
                "gt_score": safe_int(r.get("gt_score")),
            }
    return out


def load_hardcase(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def resolve_video_path(video_path: str, repo_root: Path) -> Optional[Path]:
    vp = Path(video_path)
    if vp.is_absolute() and vp.exists():
        return vp

    # common roots in this repo
    candidates = [
        repo_root / video_path,
        repo_root / "data" / "raw" / "PD4T" / "PD4T" / "PD4T" / "Videos" / video_path,
        repo_root / "data" / "raw" / "PD4T" / "PD4T" / "PD4T" / "Videos" / "Gait" / video_path,
        repo_root / "data" / "raw" / "PD4T" / "PD4T" / "PD4T" / "Videos" / "Finger tapping" / video_path,
    ]
    for c in candidates:
        if c.exists():
            return c

    # normalize clipped names: 12-104704_001_clip.mp4 -> 12-104704.mp4
    name = Path(video_path).name
    stem = Path(name).stem
    if stem.endswith("_clip"):
        stem = stem[:-5]
    stem = re.sub(r"_\d{3}$", "", stem)
    normalized = stem + Path(name).suffix

    alt_candidates = [
        repo_root / "data" / "raw" / "PD4T" / "PD4T" / "PD4T" / "Videos" / "Gait" / "038" / normalized,
        repo_root / "data" / "raw" / "PD4T" / "PD4T" / "PD4T" / "Videos" / normalized,
    ]
    for c in alt_candidates:
        if c.exists():
            return c

    # fallback expensive search by exact filename then normalized filename
    for target in [name, normalized]:
        for found in repo_root.rglob(target):
            return found
    return None


def select_20_samples(sample_map: Dict[Tuple[str, str], dict], hardcase: dict, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    finger = [v for (_, t), v in sample_map.items() if t == "finger_tapping"]
    gait = [v for (_, t), v in sample_map.items() if t == "gait"]

    # priority: hardcases first
    selected = []
    seen = set()

    for task, need in [("finger_tapping", 10), ("gait", 10)]:
        hc = hardcase.get(task, [])
        pool = finger if task == "finger_tapping" else gait
        by_id = {x["sample_id"]: x for x in pool}
        task_sel = []

        for h in hc:
            sid = h.get("sample_id", "")
            # support "finger::id" style
            sid_clean = sid.split("::", 1)[-1]
            if sid in by_id:
                task_sel.append(by_id[sid])
            elif sid_clean in by_id:
                task_sel.append(by_id[sid_clean])

        rng.shuffle(pool)
        for p in pool:
            if len(task_sel) >= need:
                break
            if p["sample_id"] not in {x["sample_id"] for x in task_sel}:
                task_sel.append(p)

        selected.extend(task_sel[:need])

    # de-dup
    out = []
    for s in selected:
        key = (s["sample_id"], s["task"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:20]


def parse_json_answer(text: str) -> Tuple[Optional[int], bool, str]:
    if not text:
        return None, False, ""

    # 1) strict JSON object parse first
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = safe_int(obj.get("answer"))
            return ans, ans is not None, obj.get("rationale", "")
        except Exception:
            pass

    # 2) fallback: plain integer answer in text
    m2 = re.search(r"\b([0-4])\b", text)
    if m2:
        return int(m2.group(1)), True, ""

    return None, False, text


def mode_ratio(scores: List[int]) -> float:
    if not scores:
        return 0.0
    from collections import Counter
    c = Counter(scores)
    return c.most_common(1)[0][1] / len(scores)


def weighted_fusion(s1: int, s2: int, gap: int) -> Tuple[int, bool]:
    final = round(s1 * 0.4 + s2 * 0.6)
    recheck = False
    if abs(s1 - s2) >= gap:
        final = max(s1, s2) - 1
        recheck = True
    return final, recheck


def has_marked_evidence(rationale: str) -> bool:
    t = (rationale or "").lower()
    keywords = ["sustained", "frequent", "arrest", "hesitation", "marked", "significant", "substantial"]
    return any(k in t for k in keywords)


def run_live_eval(samples: List[dict], repo_root: Path, out_dir: Path):
    # lazy import so script still works for selection-only mode
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=api_key)

    stage1_prompt_t = (
        "Task: {task}. Score 0-4 conservatively. "
        "Return ONLY JSON: {{\"answer\": <0-4 integer>, \"rationale\": \"short\"}}."
    )
    stage2_prompt = (
        "Task: finger_tapping specialist re-read (clinical conservative rubric). "
        "Use sustained evidence, not isolated moments. "
        "Return ONLY JSON: {\"answer\": <0-4 integer>, \"rationale\": \"short with temporal cue\"}."
    )

    def gen_text(uploaded, prompt: str, retries: int = 5) -> str:
        last_err = None
        for i in range(retries):
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[uploaded, prompt],
                    config={"max_output_tokens": 1024, "temperature": 0},
                )
                return resp.text or ""
            except Exception as e:
                last_err = e
                time.sleep(min(20, 2 * (i + 1)))
        raise last_err

    rows = []
    for s in samples:
        try:
            vp = resolve_video_path(s["video_path"], repo_root)
            if not vp:
                rows.append({**s, "status": "video_not_found"})
                continue

            uploaded = client.files.upload(file=str(vp))
            # wait active
            active = False
            for _ in range(90):
                st = str(client.files.get(name=uploaded.name).state)
                if "ACTIVE" in st:
                    active = True
                    break
                if "FAILED" in st:
                    break
                time.sleep(2)
            if not active:
                rows.append({**s, "status": "file_not_active"})
                continue

            # baseline: single stage1
            btxt = gen_text(uploaded, stage1_prompt_t.format(task=s["task"]))
            b_score, b_parse, b_rat = parse_json_answer(btxt)

            # stage1 repeated x2 for orchestration signals
            scores = []
            parses = []
            stage1_parse_failures = 0
            stage1_rationales = []
            for _ in range(2):
                txt = gen_text(uploaded, stage1_prompt_t.format(task=s["task"]))
                sc, ok, rat = parse_json_answer(txt)
                if sc is not None:
                    scores.append(sc)
                if rat:
                    stage1_rationales.append(rat)
                if not ok:
                    stage1_parse_failures += 1
                parses.append(ok)

            s1 = scores[0] if scores else None
            s2 = None
            s2_parse_ok = False
            s2_rationale = ""
            stage2_parse_failures = 0
            if s["task"] == "finger_tapping":
                # stage2 retry loop (critical for hard-cases)
                for _ in range(3):
                    t2 = gen_text(uploaded, stage2_prompt)
                    s2_try, ok2, rat2 = parse_json_answer(t2)
                    if ok2 and s2_try is not None:
                        s2 = s2_try
                        s2_parse_ok = True
                        s2_rationale = rat2
                        break
                    stage2_parse_failures += 1

            # compute A/B
            result = {
                **s,
                "status": "ok",
                "baseline_score": b_score,
                "baseline_parse_ok": b_parse,
                "baseline_rationale": b_rat,
                "stage1_scores": scores,
                "stage1_parse_ok": all(parses) if parses else False,
                "stage1_parse_failures": stage1_parse_failures,
                "stage1_rationales": stage1_rationales,
                "sampling_mode_ratio": mode_ratio(scores),
                "stage2_score": s2,
                "stage2_parse_ok": s2_parse_ok,
                "stage2_parse_failures": stage2_parse_failures,
                "stage2_rationale": s2_rationale,
            }

            for cfg in [SET_A, SET_B]:
                recheck = False
                final = s1
                if s1 is None:
                    final = None
                    recheck = True
                else:
                    unstable = mode_ratio(scores) < cfg.sampling_mode_ratio_threshold
                    if unstable:
                        recheck = True
                    if s2 is not None:
                        final, dg = weighted_fusion(s1, s2, cfg.disagreement_guardrail_gap)
                        recheck = recheck or dg
                    elif s.get("task") == "finger_tapping":
                        # no stage2 for finger => force recheck
                        recheck = True

                # finger overscoring guardrails
                if s.get("task") == "finger_tapping" and final is not None:
                    # if stage1 is high but specialist is lower, clamp
                    if s1 is not None and s2 is not None and s1 >= 3 and s2 <= 2:
                        final = min(final, 2)
                        recheck = True
                    # >=3 requires marked/sustained evidence text
                    rationale_src = (s2_rationale or " ".join(stage1_rationales)).strip()
                    if final >= 3 and not has_marked_evidence(rationale_src):
                        final = 2
                        recheck = True

                result[f"{cfg.name}_final"] = final
                result[f"{cfg.name}_recheck"] = recheck
                result[f"{cfg.name}_match"] = (final == s.get("gt_score")) if final is not None and s.get("gt_score") is not None else False

            result["baseline_match"] = (b_score == s.get("gt_score")) if b_score is not None and s.get("gt_score") is not None else False
            rows.append(result)

        except Exception as e:
            rows.append({**s, "status": "error", "error": str(e)})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary csv
    fields = [
        "sample_id", "task", "gt_score",
        "baseline_score", "baseline_parse_ok", "baseline_match",
        "stage1_parse_failures", "stage2_parse_ok", "stage2_parse_failures",
        "stage2_score", "sampling_mode_ratio",
        "set_a_final", "set_a_recheck", "set_a_match",
        "set_b_final", "set_b_recheck", "set_b_match",
    ]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # aggregate markdown
    def pct(n, d):
        return round((n / d) * 100, 1) if d else 0.0

    n = len(rows)
    b_match = sum(1 for r in rows if r.get("baseline_match"))
    a_match = sum(1 for r in rows if r.get("set_a_match"))
    b2_match = sum(1 for r in rows if r.get("set_b_match"))
    a_re = sum(1 for r in rows if r.get("set_a_recheck"))
    b_re = sum(1 for r in rows if r.get("set_b_recheck"))

    md = []
    md.append("# Orchestration v0.2 Eval Summary")
    md.append("")
    md.append(f"- Samples: {n}")
    md.append(f"- Baseline match: {b_match}/{n} ({pct(b_match,n)}%)")
    md.append(f"- Set A match: {a_match}/{n} ({pct(a_match,n)}%)")
    md.append(f"- Set B match: {b2_match}/{n} ({pct(b2_match,n)}%)")
    md.append(f"- Set A recheck rate: {a_re}/{n} ({pct(a_re,n)}%)")
    md.append(f"- Set B recheck rate: {b_re}/{n} ({pct(b_re,n)}%)")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", type=Path, required=True)
    ap.add_argument("--hardcase", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select-only", action="store_true", help="Only output selected 20 samples")
    args = ap.parse_args()

    rows = load_tracker(args.tracker)
    sample_map = build_sample_map(rows)
    hardcase = load_hardcase(args.hardcase)
    samples = select_20_samples(sample_map, hardcase, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selected_20_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.select_only:
        print(f"Selected {len(samples)} samples -> {args.out/'selected_20_samples.json'}")
        return

    run_live_eval(samples, args.repo_root, args.out)
    print(f"Done. Outputs in {args.out}")


if __name__ == "__main__":
    main()
