"""Run on ICT next to the VLM prediction files: compact the primitive predictions for transfer.

Writes prim_compact.txt: one line per clip "<source> <key> <arm><trunk><step><postural>" (0-3 each; a missing
or unparsable value counts as 0, as in training), test from the all-train adapter (prim_all_test.jsonl) and
train out-of-fold from the fold2 adapters (prim_f0_pred_fold1.jsonl, prim_f1_pred_fold0.jsonl), plus
prim_sha.txt with the sha256 of the three raw files and of the compact file. Copy prim_compact.txt to
$PD4T_TURN_ROOT and check its sha256 against prim_sha.txt before running primitive_fusion_eval.py.
The compact file holds model predictions on PD4T clips: private, never commit it.
usage: python dump_vlm_primitives.py <turnvlm dir> <out dir>
"""
import sys, os, json, hashlib

P = ["arm_swing_asymmetry", "trunk_flexion", "step_length_asymmetry", "postural_instability"]
SRC = ["prim_all_test.jsonl", "prim_f0_pred_fold1.jsonl", "prim_f1_pred_fold0.jsonl"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

src_dir, out_dir = sys.argv[1], sys.argv[2]
lines = []
for f in SRC:
    for l in open(os.path.join(src_dir, f)):
        d = json.loads(l); p = d.get("pred") or {}
        lines.append(f"{f.split('.')[0]} {d['key']} " + "".join(str(int(p.get(k) or 0)) for k in P))
out = os.path.join(out_dir, "prim_compact.txt")
open(out, "w").write("\n".join(lines) + "\n")
with open(os.path.join(out_dir, "prim_sha.txt"), "w") as fh:
    for f in SRC:
        fh.write(f"{sha(os.path.join(src_dir, f))}  {f}\n")
    fh.write(f"{sha(out)}  prim_compact.txt\n")
print(len(lines), "lines ->", out)
