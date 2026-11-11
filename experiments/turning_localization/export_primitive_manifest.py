"""Export the per-clip manifest used by the RGB early-fusion primitive model (qwen_primitive_finetune.py).

One JSON list over all 426 clips: key, pid, visit, split (official), subject, clip_dur, clinician primitives
(the four RGB-assisted ones), and the automatic pose measurements that go into the prompt as text
(gait_feats from primitive_bottleneck.py). Also assigns a deterministic 2-fold subject split inside the
official train set for out-of-fold predictions (sorted subject ids, alternating).
Output: $PD4T_TURN_ROOT/primitive_manifest.json (private; no video or pose arrays).
"""
import os, sys, json, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from primitive_bottleneck import gait_feats
from config import CLIP_ROOT

RGB_PRIMS = ["arm_swing_asymmetry", "trunk_flexion", "step_length_asymmetry", "postural_instability"]
FEAT_NAMES = ["arm_swing_amp_left", "arm_swing_amp_right", "arm_swing_asym_ratio", "ankle_gap_std",
              "trunk_lean_deg_median", "hip_vertical_bob", "progression_speed_proxy"]


def main():
    raw = {}
    for f in glob.glob(os.path.join(T.LABEL_DIR, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); raw[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    data = T.load()
    tr_subs = sorted({d["subj"] for d in data if d["split"] == "train"})
    fold = {s: i % 2 for i, s in enumerate(tr_subs)}
    out = []
    for d in data:
        v, p = d["key"].rsplit("_", 1)
        clip = os.path.join(CLIP_ROOT, p, f"{v}_{p}_clip.mp4")
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", clip]).decode())
        g = gait_feats(d["key"])
        prims = raw[d["key"]]["primitives"]
        out.append({"key": d["key"], "pid": p, "visit": v, "split": d["split"], "subject": d["subj"],
                    "fold2": fold.get(d["subj"]), "clip_dur": dur,
                    "primitives": {k: prims.get(k) for k in RGB_PRIMS},
                    "pose": {n: round(float(x), 4) for n, x in zip(FEAT_NAMES, g)}})
    dst = os.path.join(T.DATA_ROOT, "primitive_manifest.json")
    json.dump(out, open(dst, "w"), indent=0)
    print(dst, len(out), sum(o["split"] == "train" for o in out))


if __name__ == "__main__":
    main()
