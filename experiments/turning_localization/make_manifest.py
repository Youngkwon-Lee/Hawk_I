"""Reproducibility manifest: hashes of every input the results depend on + software versions + git SHA.

Writes repro/input_manifest.json (safe to commit: hashes, counts and versions only, no data).
- labels (gait_export *.jsonl), official split CSVs, per-clip sha256 of the 426 labelled 20-s clips
  (one combined digest + count), pose npz count + combined digest, checkpoints, key result files
- python / numpy / torch / sklearn / mediapipe / scipy / opencv versions, ffmpeg version line
usage: python make_manifest.py  (run on the home desktop where the data live)
"""
import os, sys, json, glob, hashlib, platform, subprocess, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_ROOT, CLIP_ROOT, LABEL_DIR, OFFICIAL_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def combined(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.relpath(p).encode() if False else os.path.basename(p).encode()); h.update(sha(p).encode())
    return h.hexdigest()


def ver(mod):
    try:
        m = importlib.import_module(mod); return getattr(m, "__version__", "?")
    except Exception as e:
        return f"not importable ({type(e).__name__})"


def main():
    out = {"generated_by": "training/make_manifest.py", "host": platform.node(), "python": sys.version.split()[0]}
    out["packages"] = {m: ver(m) for m in ("numpy", "scipy", "torch", "sklearn", "mediapipe", "cv2", "pandas")}
    try:
        out["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0]
    except Exception:
        out["ffmpeg"] = "not found"
    try:
        out["git_sha"] = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        out["git_sha"] = "not a git checkout (copy of the repo at the commit recorded in repro/README)"
    out["labels"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(LABEL_DIR, "*.jsonl")))}
    out["official_split"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(OFFICIAL_DIR, "*.csv")))}
    clips = glob.glob(os.path.join(CLIP_ROOT, "*", "*_clip.mp4"))
    out["clips_20s"] = {"count": len(clips), "combined_sha256": combined(clips)}
    npz = glob.glob(os.path.join(DATA_ROOT, "npz", "*.npz"))
    out["pose_npz"] = {"count": len(npz), "combined_sha256": combined(npz),
                       "note": "MediaPipe pose_landmarker_full, 15 fps target; regenerate with extract_pose.py"}
    task = os.path.join(DATA_ROOT, "pose_landmarker_full.task")
    if os.path.exists(task):
        out["mediapipe_model"] = sha(task)
    out["checkpoints"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(HERE, "..", "models", "*.pt")))}
    res = {}
    for name in ("eval_report.json", "sanity_20261111.json", "yourturn_baseline.json", "updrs_from_turns.json",
                 "fusion_c0b.json", "primitive_bottleneck.json", "primitive_manifest.json",
                 "primitive_bottleneck_cache.npz", "primitive_fusion_eval.json", "prim_compact.txt",
                 "tulip/tulip_turns.json", "tulip/tulip_bottleneck.json"):
        p = os.path.join(DATA_ROOT, name)
        if os.path.exists(p):
            res[name] = sha(p)
    for p in sorted(glob.glob(os.path.join(DATA_ROOT, "*test116.jsonl"))):
        res[os.path.basename(p)] = sha(p)
    out["result_files"] = res
    # VLM primitive predictions live on ICT (raw jsonl); sha256 recorded at transfer (prim_sha.txt)
    out["ict_vlm_primitive_predictions"] = {
        "prim_all_test.jsonl": "8e8a7605f3968a75fe073738007326c870a5bd3a340cb4c3fdce3599ff26933b",
        "prim_f0_pred_fold1.jsonl": "3c4dc11109480960fc1f57d7eda11edf4377fc79e2384780dc006e79aeb97888",
        "prim_f1_pred_fold0.jsonl": "d25ca216faea9c6e114458455a18a234b27949424ccdbf09876663d053f9db35",
        "prim_compact.txt": "f3a4a202d61e90bd30f358ce306e8350fd93bd5b718792b6859f717ebd3ac3db"}
    os.makedirs(os.path.join(HERE, "..", "repro"), exist_ok=True)
    dst = os.path.join(HERE, "..", "repro", "input_manifest.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(dst); print(json.dumps({k: v for k, v in out.items() if k in ("packages", "clips_20s", "pose_npz", "git_sha")}, indent=1))


if __name__ == "__main__":
    main()
