"""Does RGB+pose early fusion (Qwen3-VL QLoRA, 4 weak primitives) improve the pose bottleneck (QWK 0.743)?

Inputs ($PD4T_TURN_ROOT): primitive_bottleneck_cache.npz (pose bottleneck: Ptr_oof, Pte_hat, ytr, yte),
primitive_manifest.json (labels, fold2), prim_compact.txt (VLM predictions: test from the all-train adapter,
train out-of-fold from the two fold2 adapters; rebuilt + sha256-verified by rebuild_prim_compact.py).
Variants (integrator = same recipe as primitive_bottleneck.py, trained on OOF train primitives):
  pose_only         : reproduces 0.743
  rgb_pose_4        : the 4 primitives replaced by VLM predictions
  best_of_train_oof : per primitive, the source with higher train-OOF QWK (choice never looks at test)
Reports per-primitive QWK (train OOF / test, clips with a missing rating excluded) and UPDRS QWK with
subject-bootstrap CIs and paired differences vs pose_only.
Caveat: pose OOF uses 5 subject folds, VLM OOF uses 2 (fold2), so VLM train predictions see less data.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primitive_bottleneck as PB
from updrs_from_turns import qwk
from config import DATA_ROOT

VP = ["arm_swing_asymmetry", "trunk_flexion", "step_length_asymmetry", "postural_instability"]


def main():
    c = np.load(os.path.join(DATA_ROOT, "primitive_bottleneck_cache.npz"), allow_pickle=True)
    prims = [str(p) for p in c["prims"]]
    ktr = [str(k) for k in c["keys_tr"]]; kte = [str(k) for k in c["keys_te"]]
    gtr = np.array([str(s) for s in c["subj_tr"]]); gte = np.array([str(s) for s in c["subj_te"]])
    ytr, yte = c["ytr"].astype(int), c["yte"].astype(int)
    man = {r["key"]: r for r in json.load(open(os.path.join(DATA_ROOT, "primitive_manifest.json")))}
    vlm = {}
    for l in open(os.path.join(DATA_ROOT, "prim_compact.txt")):
        src, k, s = l.split(); vlm[k] = [int(ch) for ch in s]
    assert all(k in vlm for k in ktr + kte)
    V = lambda keys, p: np.array([vlm[k][VP.index(p)] for k in keys])
    lab = lambda keys, p: [man[k]["primitives"].get(p) for k in keys]
    res = {"per_primitive": {}}
    choose = {}
    for p in VP:
        j = prims.index(p); r = {}
        for split, keys, Ppose in (("train_oof", ktr, c["Ptr_oof"]), ("test", kte, c["Pte_hat"])):
            L = lab(keys, p); ok = np.array([x is not None for x in L]); y = np.array([int(x or 0) for x in L])
            v = V(keys, p)
            r[split] = {"n": int(ok.sum()), "pose_qwk": float(qwk(y[ok], Ppose[ok, j].astype(int))),
                        "rgb_qwk": float(qwk(y[ok], v[ok])), "rgb_nonzero_pred": int((v[ok] > 0).sum()),
                        "clin_nonzero": int((y[ok] > 0).sum())}
        choose[p] = "rgb" if r["train_oof"]["rgb_qwk"] > r["train_oof"]["pose_qwk"] else "pose"
        r["chosen_by_train_oof"] = choose[p]; res["per_primitive"][p] = r
    us = sorted(set(gte)); rs = np.random.default_rng(0)
    draws = [np.concatenate([np.where(gte == s)[0] for s in rs.choice(us, len(us))]) for _ in range(1000)]
    preds = {}
    for name in ("pose_only", "rgb_pose_4", "best_of_train_oof"):
        Xtr, Xte = c["Ptr_oof"].copy(), c["Pte_hat"].copy()
        for p in VP:
            if name == "rgb_pose_4" or (name == "best_of_train_oof" and choose[p] == "rgb"):
                j = prims.index(p); Xtr[:, j] = V(ktr, p); Xte[:, j] = V(kte, p)
        subs = sorted(set(gtr)); folds = [subs[k::5] for k in range(5)]
        C, _ = PB.fit_cv(Xtr, ytr, gtr, folds)
        pr = PB.make_pipeline(PB.StandardScaler(), PB.LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr).predict(Xte)
        preds[name] = pr
        res[name] = {"qwk": float(qwk(yte, pr)), "acc": float((pr == yte).mean()), "mae": float(np.abs(pr - yte).mean()),
                     "qwk_95ci": [float(np.percentile([qwk(yte[i], pr[i]) for i in draws], q)) for q in (2.5, 97.5)], "C": C}
    for name in ("rgb_pose_4", "best_of_train_oof"):
        d = [qwk(yte[i], preds[name][i]) - qwk(yte[i], preds["pose_only"][i]) for i in draws]
        res[name]["delta_vs_pose_95ci"] = [float(np.percentile(d, q)) for q in (2.5, 97.5)]
        res[name]["identical_to_pose_only"] = bool((preds[name] == preds["pose_only"]).all())
    json.dump(res, open(os.path.join(DATA_ROOT, "primitive_fusion_eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
