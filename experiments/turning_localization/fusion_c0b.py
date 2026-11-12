"""Late fusion of the RGB score model (Qwen3-VL C0B QLoRA, full clip) with the turn-based pose model.

usage: python fusion_c0b.py c0b_preds.json      ([[key, score], ...] for the 116 test clips)
Fusion rule fixed a priori (no tuning on test): fused = round(0.5 * C0B score + 0.5 * expected score of the
turn model T/pipeline, i.e. sum_k k * P(k)). Reports acc / MAE / QWK with subject-bootstrap 95% CI for each
system and for the fusion, plus the paired bootstrap CI of (fusion - C0B) QWK.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from updrs_from_turns import scores, qwk


def metrics(y, p, te, draws):
    bq = [qwk(y[i], p[i]) for i in draws]
    return {"acc": float((p == y).mean()), "mae": float(np.abs(p - y).mean()), "qwk": float(qwk(y, p)),
            "qwk_95ci": [float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))]}


def main(path):
    te = [d for d in T.load() if d["split"] == "test"]
    sc = scores("test"); y = np.array([sc[d["key"]] for d in te])
    c0b = dict((k, v) for k, v in json.load(open(path)))
    pc = np.array([c0b[d["key"]] for d in te])
    proba = {w: np.load(os.path.join(T.DATA_ROOT, f"updrs_{w}_pipeline_proba.npy")) for w in ("T", "TP")}
    tsubs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0)
    draws = [[i for s in rs.choice(tsubs, len(tsubs), replace=True) for i, d in enumerate(te) if d["subj"] == s]
             for _ in range(1000)]
    out = {"C0B_rgb": metrics(y, pc, te, draws)}
    for w, P in proba.items():
        k = np.arange(P.shape[1]); ev = P @ k; pt = P.argmax(1)
        fused = np.clip(np.rint(0.5 * pc + 0.5 * ev), 0, 4).astype(int)
        out[f"turn_{w}"] = metrics(y, pt, te, draws)
        out[f"fusion_C0B+turn_{w}"] = metrics(y, fused, te, draws)
        diff = [qwk(y[i], fused[i]) - qwk(y[i], pc[i]) for i in draws]
        out[f"fusion_C0B+turn_{w}"]["qwk_minus_C0B_95ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(T.DATA_ROOT, "fusion_c0b.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
