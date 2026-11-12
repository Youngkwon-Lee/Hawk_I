"""Predict the PD4T gait score (MDS-UPDRS 3.10, official labels) from far-end turn information only.

Feature sets (per clip):
  T  turn timing: n far turns, mean/max turn duration, mean rotation speed (Your Turn angle / duration)
  P  pose during turns: mean and std of the 76 BiGRU input features inside the turn segments
  TP both
Turn segments: TRAIN uses the clinician intervals; TEST uses either the clinician intervals ("oracle") or the
BiGRU seed-100 predictions ("pipeline", the realistic setting). Clips without a turn get zeros + a flag.
Classifier: standardised multinomial logistic regression; C chosen by 5-fold subject-grouped CV on train
(macro QWK). Test reported once: exact accuracy, MAE, QWK (+ subject-bootstrap 95% CI for QWK).
Reference on the same split: Qwen3-VL C0B QLoRA on full RGB clips, QWK 0.621 / MAE 0.40 / acc 0.64.
"""
import os, sys, json, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from eval_report import model_segs, MODELS
from yourturn_baseline import yourturn_angle


def scores(split):
    out = {}
    for r in csv.reader(open(os.path.join(T.OFFICIAL_DIR, f"{split}.csv"))):
        v, p = r[0].rsplit("_", 1); out[f"{v}_{p}"] = int(float(r[2]))
    return out


def feats(d, segs):
    t, x = d["t"], d["x"]
    if not segs:
        return np.zeros(4 + 1), np.zeros(2 * x.shape[1])
    durs = [e - s for s, e in segs]
    speed = [yourturn_angle(d["key"], s, e) / max(e - s, 1e-3) for s, e in segs]
    tf = np.array([len(segs), np.mean(durs), np.max(durs), np.mean(speed), 1.0])
    m = np.zeros(len(t), bool)
    for s, e in segs:
        m |= (t >= s) & (t <= e)
    if m.sum() < 2:
        m[:] = True
    pf = np.concatenate([x[m].mean(0), x[m].std(0)])
    return tf, pf


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3, 4])


def build(items, seg_list, which):
    rows = []
    for d, segs in zip(items, seg_list):
        tf, pf = feats(d, [(s[0], s[1]) for s in segs])
        rows.append({"T": tf, "P": pf, "TP": np.concatenate([tf, pf])}[which])
    return np.array(rows)


def main():
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    sc = {**scores("train"), **scores("test")}
    ytr = np.array([sc[d["key"]] for d in tr]); yte = np.array([sc[d["key"]] for d in te])
    lab_tr = [[(L["start"], L["end"]) for L in d["lab"]] for d in tr]
    lab_te = [[(L["start"], L["end"]) for L in d["lab"]] for d in te]
    pred_te = model_segs(os.path.join(MODELS, "turn_bigru_seed100.pt"), te)
    subs = sorted({d["subj"] for d in tr}); folds = [subs[k::5] for k in range(5)]
    res = {"score_dist_train": np.bincount(ytr, minlength=5).tolist(), "score_dist_test": np.bincount(yte, minlength=5).tolist(),
           "majority_class_test": {"acc": float((yte == np.bincount(ytr).argmax()).mean()),
                                   "mae": float(np.abs(yte - np.bincount(ytr).argmax()).mean()), "qwk": 0.0}}
    for which in ("T", "P", "TP"):
        Xtr = build(tr, lab_tr, which)
        best, bestq = None, -9
        for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            pr = np.zeros_like(ytr)
            for fs in folds:
                vi = np.array([d["subj"] in fs for d in tr])
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
                clf.fit(Xtr[~vi], ytr[~vi]); pr[vi] = clf.predict(Xtr[vi])
            q = qwk(ytr, pr)
            if q > bestq:
                best, bestq = C, q
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
        for mode, segs in (("oracle_turns", lab_te), ("pipeline_turns", pred_te)):
            p = clf.predict(build(te, segs, which))
            rs = np.random.default_rng(0); tsubs = sorted({d["subj"] for d in te}); bq = []
            for _ in range(1000):
                idx = [i for s in rs.choice(tsubs, len(tsubs), replace=True) for i, d in enumerate(te) if d["subj"] == s]
                bq.append(qwk(yte[idx], p[idx]))
            res[f"{which}_{mode}"] = {"C": best, "cv_qwk": float(bestq), "acc": float((p == yte).mean()),
                                      "mae": float(np.abs(p - yte).mean()), "qwk": float(qwk(yte, p)),
                                      "qwk_95ci": [float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))]}
        np.save(os.path.join(T.DATA_ROOT, f"updrs_{which}_pipeline_proba.npy"),
                clf.predict_proba(build(te, pred_te, which)))
    json.dump(res, open(os.path.join(T.DATA_ROOT, "updrs_from_turns.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
