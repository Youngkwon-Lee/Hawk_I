"""Paired subject-bootstrap comparison: BiGRU checkpoints vs a VLM prediction file (same 116 test clips).

usage: python compare_paired.py vlm_predictions.jsonl
Reports F1@tIoU0.3/0.5/0.7 for each BiGRU seed and the VLM, and the 95% CI of (BiGRU - VLM) for each seed,
resampling the 8 test subjects with replacement (1000 draws, same draws for both systems).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from eval_report import tiou_f1, model_segs, MODELS, SEEDS


def main(path):
    te = [d for d in T.load() if d["split"] == "test"]
    vlm = {json.loads(l)["key"]: json.loads(l)["pred"] for l in open(path)}
    v_of = {d["key"]: vlm.get(d["key"], []) for d in te}
    subs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0)
    draws = [rs.choice(subs, len(subs), replace=True) for _ in range(1000)]
    out = {"vlm_file": os.path.basename(path)}
    for s in SEEDS:
        segs = model_segs(os.path.join(MODELS, f"turn_bigru_seed{s}.pt"), te)
        b_of = {d["key"]: g for d, g in zip(te, segs)}
        for thr in (0.3, 0.5, 0.7):
            diff = []
            for dr in draws:
                items = [d for x in dr for d in te if d["subj"] == x]
                diff.append(tiou_f1([b_of[d["key"]] for d in items], items, thr)
                            - tiou_f1([v_of[d["key"]] for d in items], items, thr))
            out[f"seed{s}_tiou{thr}"] = {
                "bigru": tiou_f1([b_of[d["key"]] for d in te], te, thr),
                "vlm": tiou_f1([v_of[d["key"]] for d in te], te, thr),
                "diff_bigru_minus_vlm_95ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(path + ".paired.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
