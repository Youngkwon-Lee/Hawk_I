"""Turn duration agreement on matched turns (one-to-one tIoU>=0.1 matching), PD4T official test.

usage: python duration_agreement.py [vlm_predictions.jsonl ...]
For the BiGRU (seed 100) and each VLM file: n matched turns, duration MAE (s), Pearson r, ICC(2,1) absolute
agreement, mean bias (pred - label). Also Spearman of per-clip mean label/pred duration with the PD4T gait score.
"""
import os, sys, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from eval_report import model_segs, MODELS


def icc21(a, b):
    x = np.stack([a, b], 1); n, k = x.shape; m = x.mean()
    msr = k * ((x.mean(1) - m) ** 2).sum() / (n - 1); msc = n * ((x.mean(0) - m) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def match(te, segs, thr=0.1):
    L, P, keys = [], [], []
    for d, ss in zip(te, segs):
        pairs = sorted(((T.tiou((l["start"], l["end"]), s), i, j) for i, l in enumerate(d["lab"])
                        for j, s in enumerate(ss)), reverse=True)
        ul, us = set(), set()
        for v, i, j in pairs:
            if v >= thr and i not in ul and j not in us:
                ul.add(i); us.add(j); l = d["lab"][i]; s = ss[j]
                L.append(l["end"] - l["start"]); P.append(s[1] - s[0]); keys.append(d["key"])
    return np.array(L), np.array(P), keys


def summary(name, te, segs, score):
    L, P, keys = match(te, segs)
    out = {"n_matched": len(L), "n_labels": sum(len(d["lab"]) for d in te),
           "label_dur_mean_s": float(L.mean()), "pred_dur_mean_s": float(P.mean()),
           "mae_s": float(np.abs(P - L).mean()), "bias_s": float((P - L).mean()),
           "pearson_r": float(pearsonr(L, P)[0]), "icc21": icc21(L, P)}
    per = {}
    for k, p in zip(keys, P):
        per.setdefault(k, []).append(p)
    ks = [k for k in per if k in score]
    out["spearman_pred_dur_vs_pd4t_score"] = float(spearmanr([np.mean(per[k]) for k in ks], [score[k] for k in ks])[0])
    return name, out


def main(vlm_files):
    te = [d for d in T.load() if d["split"] == "test"]
    import csv
    score = {}
    for r in csv.reader(open(os.path.join(T.OFFICIAL_DIR, "test.csv"))):
        v, p = r[0].rsplit("_", 1); score[f"{v}_{p}"] = float(r[2])
    res = {}
    labdur = {d["key"]: np.mean([l["end"] - l["start"] for l in d["lab"]]) for d in te if d["lab"]}
    ks = [k for k in labdur if k in score]
    res["label_only"] = {"spearman_label_dur_vs_pd4t_score": float(spearmanr([labdur[k] for k in ks], [score[k] for k in ks])[0]),
                         "n_clips": len(ks)}
    n, o = summary("BiGRU seed100", te, model_segs(os.path.join(MODELS, "turn_bigru_seed100.pt"), te), score); res[n] = o
    for f in vlm_files:
        pr = {json.loads(l)["key"]: json.loads(l)["pred"] for l in open(f)}
        n, o = summary(os.path.basename(f), te, [pr.get(d["key"], []) for d in te], score); res[n] = o
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
