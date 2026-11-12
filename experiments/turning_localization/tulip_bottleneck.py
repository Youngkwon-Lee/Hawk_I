"""External validation of the automatic primitive bottleneck (PD4T QWK 0.743) on TULIP Camera1 gait.

Nothing is fitted on TULIP labels. Steps:
 1. Rebuild the bottleneck exactly as primitive_bottleneck.py (same features, OOF turn models seeds 200-204,
    C by 5-fold subject CV, oof-trained integrator) and check it reproduces the PD4T test QWK.
 2. TULIP: each 60-s Camera1 video is cut into three 20-s windows (PD4T clips are 20 s). Turns = frozen
    BiGRU seed-100 intervals (tulip_turns.json) whose centre falls in the window, absolute times kept.
    Window features -> 7 primitives -> expected UPDRS 3.10 score; subject score = mean of the 3 windows.
 3. Variants
    A frozen      : PD4T-trained models applied as is
    B domain-std  : label-free feature alignment - each stage-1 feature is z-scored with TULIP's own
                    window statistics and mapped onto PD4T-train mean/sd (no TULIP labels used)
    T turn-only   : turn-timing features only (updrs_from_turns.py set T), frozen - reference
 4. Compare with the 3 clinicians' gait scores: Spearman (pred vs clinician mean), MAE, QWK and exact
    agreement vs rounded clinician mean, PD vs HT separation; clinicians' own pairwise agreement.
Note: TULIP pose is 16 fps vs PD4T 15 fps (frame-difference features differ by ~7%).
usage: python tulip_bottleneck.py  -> $PD4T_TURN_ROOT/tulip/tulip_bottleneck.json (private)
"""
import os, sys, json, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
import primitive_bottleneck as PB
from updrs_from_turns import scores, qwk, feats as turn_feats
from eval_report import model_segs, MODELS

TULIP = os.environ.get("TULIP_ROOT", "/mnt/d/1_TULIP")
OUT = os.path.join(T.DATA_ROOT, "tulip")
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def gait_feats_arr(img, t):
    Tn = len(t)
    xy = T.interp(img[:, :, :2].reshape(Tn, -1)).reshape(Tn, 33, 2)
    hip = (xy[:, 23] + xy[:, 24]) / 2; sh = (xy[:, 11] + xy[:, 12]) / 2
    tor = np.linalg.norm(sh - hip, axis=1) + 1e-3
    arm = lambda w: np.percentile((xy[:, w, 1] - hip[:, 1]) / tor, 90) - np.percentile((xy[:, w, 1] - hip[:, 1]) / tor, 10)
    aL, aR = arm(15), arm(16)
    gap = np.abs(xy[:, 27, 1] - xy[:, 28, 1]) / tor
    lean = np.degrees(np.arctan2(np.abs(sh[:, 0] - hip[:, 0]), np.abs(hip[:, 1] - sh[:, 1]) + 1e-3))
    bob = np.std(np.diff(hip[:, 1]) / tor[1:])
    speed = np.median(np.abs(np.gradient(np.log(tor), t)))
    return np.array([aL, aR, abs(aL - aR) / (aL + aR + 1e-3), np.std(gap), np.median(lean), bob, speed])


def window_feats(key, raw, x, t, segs):
    x_, = [x]
    summ = np.concatenate([x_.mean(0), x_.std(0), np.percentile(x_, 10, 0), np.percentile(x_, 90, 0)])
    d = {"key": key, "t": t, "x": x}
    return np.concatenate([summ, gait_feats_arr(raw["img"], raw["t"]), turn_feats(d, segs)[0]]), turn_feats(d, segs)[0]


def main():
    # ---------- 1. rebuild PD4T bottleneck ----------
    raw = {}
    for f in glob.glob(os.path.join(T.LABEL_DIR, "*.jsonl")):
        for l in open(f):
            r = json.loads(l); raw[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    sc = {**scores("train"), **scores("test")}
    ytr = np.array([sc[d["key"]] for d in tr]); yte = np.array([sc[d["key"]] for d in te])
    gtr = np.array([d["subj"] for d in tr]); subs = sorted(set(gtr)); folds = [subs[k::5] for k in range(5)]
    seg_te = [[(s[0], s[1]) for s in ss] for ss in model_segs(os.path.join(MODELS, "turn_bigru_seed100.pt"), te)]
    seg_tr = [None] * len(tr)
    for k, fs in enumerate(folds):
        trn = [d for d in tr if d["subj"] not in fs]; vi = [i for i, d in enumerate(tr) if d["subj"] in fs]
        net, mu, sd, _ = T.train(list(trn), 40, seed=200 + k)
        P = T.predict(net, mu, sd, [tr[i] for i in vi])
        for i, p in zip(vi, P):
            seg_tr[i] = [(s[0], s[1]) for s in T.segments(p, tr[i]["t"], 0.6)]
    Xtr = np.array([PB.clip_feats(d, s) for d, s in zip(tr, seg_tr)]); Xte = np.array([PB.clip_feats(d, s) for d, s in zip(te, seg_te)])
    prim = lambda d, k: raw[d["key"]]["primitives"].get(k)
    Ptr_oof = np.zeros((len(tr), len(PB.PRIMS))); Pte_hat = np.zeros((len(te), len(PB.PRIMS)))
    stage1 = []
    for j, k in enumerate(PB.PRIMS):
        ok = np.array([prim(d, k) is not None for d in tr]); y = np.array([float(prim(d, k) or 0) for d in tr]).astype(int)
        C, _ = PB.fit_cv(Xtr[ok], y[ok], gtr[ok], folds)
        for fs in folds:
            vi = np.isin(gtr, fs); Ptr_oof[vi, j] = LR(C).fit(Xtr[ok & ~vi], y[ok & ~vi]).predict(Xtr[vi])
        m = LR(C).fit(Xtr[ok], y[ok]); stage1.append(m); Pte_hat[:, j] = m.predict(Xte)
    C2, _ = PB.fit_cv(Ptr_oof, ytr, gtr, folds); integ = LR(C2).fit(Ptr_oof, ytr)
    pd4t_qwk = float(qwk(yte, integ.predict(Pte_hat)))
    print("PD4T test QWK (should be 0.743):", round(pd4t_qwk, 3), flush=True)
    # turn-only reference model (set T, as tulip_external.updrs)
    Ttr = np.array([turn_feats(d, [(L["start"], L["end"]) for L in d["lab"]])[0] for d in tr])
    tmodel = LR(3.0).fit(Ttr, ytr)

    # ---------- 2. TULIP windows ----------
    import pandas as pd
    turns = json.load(open(os.path.join(OUT, "tulip_turns.json")))
    lab = pd.read_excel(os.path.join(TULIP, "Final_Label_TULIPver1.0_CVPR_2025_Mar.xlsx"), header=None)
    it = lab.iloc[2:, :5]; it.columns = ["n", "item", "c1", "c2", "c3"]
    dx = lab.iloc[3:, 6:10]; dx.columns = ["n", "d1", "d2", "d3"]
    T_NPZ = T.NPZ; T.NPZ = os.path.join(OUT, "npz")   # yourturn_angle reads pose by key
    W = []   # (subject, window, X, Tfeat)
    for f in sorted(glob.glob(os.path.join(OUT, "npz", "*.npz"))):
        key = os.path.basename(f)[:-4]; d = np.load(f); x, t = T.features(d)
        segs = [(s["start"], s["end"]) for s in turns[key]["seed100"]]
        for w0 in (0, 20, 40):
            m = (t >= w0) & (t < w0 + 20)
            ws = [(s, e) for s, e in segs if w0 <= (s + e) / 2 < w0 + 20]
            X, tf = window_feats(key, {"img": d["img"][m], "t": d["t"][m]}, x[m], t[m], ws)
            W.append((key, w0, X, tf))
    T.NPZ = T_NPZ
    Xw = np.array([w[2] for w in W]); Tw = np.array([w[3] for w in W])
    mu_tr, sd_tr = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xw_B = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-9) * sd_tr + mu_tr
    cls = np.arange(5)
    def run(X):
        P = np.column_stack([m.predict(X) for m in stage1])
        pr = integ.predict_proba(P); ev = pr @ integ.classes_
        return P, ev
    out_rows = {}
    variants = {}
    for name, X in (("A_frozen", Xw), ("B_domain_std", Xw_B)):
        P, ev = run(X); variants[name] = (P, ev)
    variants["T_turn_only"] = (None, tmodel.predict_proba(Tw) @ tmodel.classes_)
    keys = sorted({w[0] for w in W})
    rows = []
    for k in keys:
        n = int(k.split("_")[1])
        g = it[(it["n"] == n) & (it["item"] == "Gait")][["c1", "c2", "c3"]].astype(float).values[0]
        dd = dx[dx["n"] == n][["d1", "d2", "d3"]].values[0].tolist()
        idx = [i for i, w in enumerate(W) if w[0] == k]
        r = {"subject": k, "clin": g.tolist(), "clin_mean": float(g.mean()), "dx": max(set(dd), key=dd.count),
             "n_turns": len(turns[k]["seed100"]), "mean_turn_dur": float(np.mean([s["dur"] for s in turns[k]["seed100"]] or [np.nan]))}
        for name, (P, ev) in variants.items():
            r[name] = float(np.mean(ev[idx]))
            if P is not None:
                r[name + "_prims"] = dict(zip(PB.PRIMS, np.round(P[idx].mean(0), 2).tolist()))
        rows.append(r)
    clin_mean = np.array([r["clin_mean"] for r in rows]); clin_round = np.rint(clin_mean).astype(int)
    res = {"pd4t_test_qwk_rebuilt": pd4t_qwk, "n_subjects": len(rows),
           "clin_dist_rounded": np.bincount(clin_round, minlength=5).tolist()}
    C3 = np.array([r["clin"] for r in rows])
    res["clinician_pairwise_qwk"] = [float(qwk(C3[:, i].astype(int), C3[:, j].astype(int))) for i, j in ((0, 1), (0, 2), (1, 2))]
    for name in variants:
        p = np.array([r[name] for r in rows]); pr = np.clip(np.rint(p), 0, 4).astype(int)
        rho, pv = spearmanr(p, clin_mean)
        v = {"spearman": float(rho), "spearman_p": float(pv), "mae": float(np.abs(p - clin_mean).mean()),
             "qwk_vs_rounded_mean": float(qwk(clin_round, pr)), "exact": float((pr == clin_round).mean()),
             "within1": float((np.abs(pr - clin_round) <= 1).mean()), "mean_pred": float(p.mean()),
             "qwk_vs_each_clinician": [float(qwk(C3[:, c].astype(int), pr)) for c in range(3)]}
        pdv = p[[r["dx"] == "PD" for r in rows]]; htv = p[[r["dx"] == "HT" for r in rows]]
        if len(pdv) and len(htv):
            v["PD_mean"], v["HT_mean"] = float(pdv.mean()), float(htv.mean())
            v["PD_vs_HT_p"] = float(mannwhitneyu(pdv, htv).pvalue)
        res[name] = v
    res["rows"] = rows
    json.dump(res, open(os.path.join(OUT, "tulip_bottleneck.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r["subject"], r["dx"], r["clin"], "A", round(r["A_frozen"], 2), "B", round(r["B_domain_std"], 2),
              "T", round(r["T_turn_only"], 2), "turns", r["n_turns"], "dur", round(r["mean_turn_dur"], 2))


if __name__ == "__main__":
    main()
