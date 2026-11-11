"""Automatic primitive bottleneck: pose -> 7 clinician gait primitives -> UPDRS 3.10 (PD4T official split).

Stage 1 (primitive predictors): per clip, pose summary features (mean/std/p10/p90 of the 76 frame features of
turn_sup.features) + 7 hand-made gait features (arm-swing amplitude L/R and its asymmetry, ankle-gap
oscillation, trunk lean, hip vertical bob, progression speed proxy) + 5 automatic turn-timing features (BiGRU
seed 100 on test; out-of-fold on train). One standardised multinomial logistic regression per primitive
(0-3; clips with a missing rating are left out of that primitive's training), C by 5-fold subject-grouped CV.
freezing_of_gait is excluded (193/426 missing, 2 positives).
Stage 2 (integrator): logistic regression on the 7 primitives only. Two variants:
  oracle-trained  : trained on clinician primitives, applied to predicted test primitives
  oof-trained     : trained on out-of-fold predicted train primitives (matches test conditions)
Reports per-primitive test agreement (accuracy, QWK) and UPDRS QWK with subject-bootstrap CIs.
"""
import os, sys, json, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from updrs_from_turns import scores, qwk, feats as turn_feats
from eval_report import model_segs, MODELS

PRIMS = ["gait_speed_reduction", "shortened_stride", "step_length_asymmetry", "arm_swing_asymmetry",
         "festination", "trunk_flexion", "postural_instability"]
CS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def gait_feats(key):
    d = np.load(os.path.join(T.NPZ, key + ".npz")); Tn = len(d["t"])
    xy = T.interp(d["img"][:, :, :2].reshape(Tn, -1)).reshape(Tn, 33, 2)
    hip = (xy[:, 23] + xy[:, 24]) / 2; sh = (xy[:, 11] + xy[:, 12]) / 2
    tor = np.linalg.norm(sh - hip, axis=1) + 1e-3
    arm = lambda w: np.percentile((xy[:, w, 1] - hip[:, 1]) / tor, 90) - np.percentile((xy[:, w, 1] - hip[:, 1]) / tor, 10)
    aL, aR = arm(15), arm(16)
    gap = np.abs(xy[:, 27, 1] - xy[:, 28, 1]) / tor
    lean = np.degrees(np.arctan2(np.abs(sh[:, 0] - hip[:, 0]), np.abs(hip[:, 1] - sh[:, 1]) + 1e-3))
    bob = np.std(np.diff(hip[:, 1]) / tor[1:])
    speed = np.median(np.abs(np.gradient(np.log(tor), d["t"])))
    return np.array([aL, aR, abs(aL - aR) / (aL + aR + 1e-3), np.std(gap), np.median(lean), bob, speed])


def clip_feats(d, segs):
    x = d["x"]
    summ = np.concatenate([x.mean(0), x.std(0), np.percentile(x, 10, 0), np.percentile(x, 90, 0)])
    return np.concatenate([summ, gait_feats(d["key"]), turn_feats(d, segs)[0]])


def fit_cv(X, y, groups, folds):
    best, bq = None, -9
    for C in CS:
        pr = np.zeros_like(y)
        for fs in folds:
            vi = np.isin(groups, fs)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            m.fit(X[~vi], y[~vi]); pr[vi] = m.predict(X[vi])
        q = qwk(y, pr)
        if q > bq: best, bq = C, q
    return best, bq


def main():
    raw = {}
    for f in glob.glob(os.path.join(T.LABEL_DIR, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); raw[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    sc = {**scores("train"), **scores("test")}
    ytr = np.array([sc[d["key"]] for d in tr]); yte = np.array([sc[d["key"]] for d in te])
    gtr = np.array([d["subj"] for d in tr]); subs = sorted(set(gtr)); folds = [subs[k::5] for k in range(5)]
    # automatic turns: test from the frozen seed-100 checkpoint; train out-of-fold (fold model on 4/5 subjects)
    seg_te = [[(s[0], s[1]) for s in ss] for ss in model_segs(os.path.join(MODELS, "turn_bigru_seed100.pt"), te)]
    seg_tr = [None] * len(tr)
    for k, fs in enumerate(folds):
        trn = [d for d in tr if d["subj"] not in fs]; vi = [i for i, d in enumerate(tr) if d["subj"] in fs]
        net, mu, sd, _ = T.train(list(trn), 40, seed=200 + k)
        P = T.predict(net, mu, sd, [tr[i] for i in vi])
        for i, p in zip(vi, P):
            seg_tr[i] = [(s[0], s[1]) for s in T.segments(p, tr[i]["t"], 0.6)]
    Xtr = np.array([clip_feats(d, s) for d, s in zip(tr, seg_tr)]); Xte = np.array([clip_feats(d, s) for d, s in zip(te, seg_te)])
    prim = lambda d, k: raw[d["key"]]["primitives"].get(k)
    Ptr_clin = np.array([[float(prim(d, k) or 0) for k in PRIMS] for d in tr])
    Pte_clin = np.array([[float(prim(d, k) or 0) for k in PRIMS] for d in te])
    Ptr_oof = np.zeros_like(Ptr_clin); Pte_hat = np.zeros_like(Pte_clin)
    tsubs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0)
    draws = [[i for s in rs.choice(tsubs, len(tsubs), replace=True) for i, d in enumerate(te) if d["subj"] == s] for _ in range(1000)]
    ci = lambda y, p: [float(np.percentile([qwk(y[i], p[i]) for i in draws], q)) for q in (2.5, 97.5)]
    res = {"per_primitive_test": {}}
    for j, k in enumerate(PRIMS):
        ok = np.array([prim(d, k) is not None for d in tr]); y = Ptr_clin[:, j].astype(int)
        C, cvq = fit_cv(Xtr[ok], y[ok], gtr[ok], folds)
        for fs in folds:   # out-of-fold predictions for every train clip
            vi = np.isin(gtr, fs)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            m.fit(Xtr[ok & ~vi], y[ok & ~vi]); Ptr_oof[vi, j] = m.predict(Xtr[vi])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr[ok], y[ok])
        Pte_hat[:, j] = m.predict(Xte)
        okt = np.array([prim(d, k) is not None for d in te]); yt = Pte_clin[:, j].astype(int)
        base = int(np.bincount(y[ok]).argmax())
        res["per_primitive_test"][k] = {"n": int(okt.sum()), "nonzero_test": int((yt[okt] > 0).sum()),
            "acc": float((Pte_hat[okt, j] == yt[okt]).mean()), "majority_acc": float((yt[okt] == base).mean()),
            "qwk": float(qwk(yt[okt], Pte_hat[okt, j].astype(int))), "cv_qwk": float(cvq), "C": C}
    for name, Xi_tr in (("oracle_trained", Ptr_clin), ("oof_trained", Ptr_oof)):
        C, _ = fit_cv(Xi_tr, ytr, gtr, folds)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xi_tr, ytr)
        p = m.predict(Pte_hat)
        res[f"updrs_bottleneck_{name}"] = {"acc": float((p == yte).mean()), "mae": float(np.abs(p - yte).mean()),
                                           "qwk": float(qwk(yte, p)), "qwk_95ci": ci(yte, p)}
    C, _ = fit_cv(Xtr, ytr, gtr, folds)   # same features, no bottleneck (direct pose -> score), for reference
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
    p = m.predict(Xte)
    res["updrs_direct_pose_no_bottleneck"] = {"acc": float((p == yte).mean()), "mae": float(np.abs(p - yte).mean()),
                                              "qwk": float(qwk(yte, p)), "qwk_95ci": ci(yte, p)}
    json.dump(res, open(os.path.join(T.DATA_ROOT, "primitive_bottleneck.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
