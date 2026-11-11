"""How much of the PD4T gait score (official labels) do the clinician's gait primitives explain?

'Oracle bottleneck': the 8 clinician primitive ratings (0-3; None -> 0 plus a missing flag) are the only input to a
standardised multinomial logistic regression (C by 5-fold subject-grouped CV on train), evaluated once on the
official test split. Also: + turn timing features (clinician turn intervals). Not a deployable model (needs the
clinician's primitive ratings); it is the upper bound for a primitive-bottleneck design.
"""
import os, sys, json, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from updrs_from_turns import scores, qwk, feats

PRIMS = ["gait_speed_reduction", "shortened_stride", "step_length_asymmetry", "arm_swing_asymmetry",
         "festination", "freezing_of_gait", "trunk_flexion", "postural_instability"]


def main():
    raw = {}
    for f in glob.glob(os.path.join(T.LABEL_DIR, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); raw[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    sc = {**scores("train"), **scores("test")}
    ytr = np.array([sc[d["key"]] for d in tr]); yte = np.array([sc[d["key"]] for d in te])

    def prim(d):
        p = raw[d["key"]]["primitives"]
        return np.array([0.0 if p.get(k) is None else float(p[k]) for k in PRIMS] +
                        [float(p.get(k) is None) for k in PRIMS])
    sets = {"primitives": lambda d: prim(d),
            "primitives+turn": lambda d: np.concatenate([prim(d), feats(d, [(L["start"], L["end"]) for L in d["lab"]])[0]]),
            "annotator_own_3_10": None}
    subs = sorted({d["subj"] for d in tr}); folds = [subs[k::5] for k in range(5)]
    tsubs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0)
    draws = [[i for s in rs.choice(tsubs, len(tsubs), replace=True) for i, d in enumerate(te) if d["subj"] == s] for _ in range(1000)]
    res = {}
    for name, fn in sets.items():
        if fn is None:
            p = np.array([int(raw[d["key"]]["updrs_3_10"]) for d in te])
        else:
            Xtr = np.array([fn(d) for d in tr]); Xte = np.array([fn(d) for d in te])
            best, bq = None, -9
            for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                pr = np.zeros_like(ytr)
                for fs in folds:
                    vi = np.array([d["subj"] in fs for d in tr])
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
                    m.fit(Xtr[~vi], ytr[~vi]); pr[vi] = m.predict(Xtr[vi])
                q = qwk(ytr, pr)
                if q > bq: best, bq = C, q
            m = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
            p = m.predict(Xte)
            if name == "primitives":
                coef = m[-1].coef_; res["coef_by_class"] = {int(c): dict(zip(PRIMS, np.round(coef[i][:8], 2).tolist())) for i, c in enumerate(m[-1].classes_)}
        b = [qwk(yte[i], p[i]) for i in draws]
        res[name] = {"acc": float((p == yte).mean()), "mae": float(np.abs(p - yte).mean()), "qwk": float(qwk(yte, p)),
                     "qwk_95ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
