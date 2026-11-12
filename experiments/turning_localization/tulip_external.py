"""External validation on TULIP (CVPR 2025; 12 subjects with the MDS-UPDRS gait task, 3 clinicians).

Steps (models are frozen; nothing is trained on TULIP):
  extract  MediaPipe pose for Subject_*/25. Gait/Camera1.mp4 (frontal view, walk toward/away, 60 s, 80 fps)
  apply    BiGRU checkpoints (seeds 100-102) -> far-end turn intervals per subject; per-turn duration and
           rotation (Your Turn formula); QA contact sheets for human checking
  updrs    turn-timing model trained on PD4T train (feature set T, same recipe as updrs_from_turns.py) applied
           to three 20-s windows per subject, averaged; compared with the 3 clinicians' gait scores
usage: python tulip_external.py extract|apply|updrs
Outputs go to $PD4T_TURN_ROOT/tulip (private; no video or pose data in Git).
"""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from config import DATA_ROOT

TULIP = os.environ.get("TULIP_ROOT", "/mnt/d/1_TULIP")
OUT = os.path.join(DATA_ROOT, "tulip")
CAM = os.environ.get("TULIP_CAM", "Camera1")


def subjects():
    return sorted(glob.glob(os.path.join(TULIP, "Subject_*")))


def extract():
    import extract_pose as E
    from multiprocessing import Pool
    os.makedirs(os.path.join(OUT, "npz"), exist_ok=True)
    E.OUT = os.path.join(OUT, "npz")
    jobs = []
    for s in subjects():
        src = os.path.join(s, "25. Gait", f"{CAM}.mp4")
        link = os.path.join(OUT, f"{os.path.basename(s)}_{CAM}_clip.mp4")   # extract_pose names by *_clip.mp4
        if not os.path.exists(link):
            os.symlink(src, link)
        jobs.append(link)
    t0 = time.time()
    with Pool(4) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(E.run, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s] {name} {msg}", flush=True)


def load_tulip():
    items = []
    for f in sorted(glob.glob(os.path.join(OUT, "npz", "*.npz"))):
        d = np.load(f); x, t = T.features(d)
        items.append({"key": os.path.basename(f)[:-4], "x": x, "t": t, "y": np.zeros(len(t), np.float32),
                      "lab": [], "npose": d["npose"]})
    return items


def apply():
    import torch
    from yourturn_baseline import yourturn_angle
    items = load_tulip()
    res = {}
    for seed in (100, 101, 102):
        ck = torch.load(os.path.join(os.path.dirname(__file__), "..", "models", f"turn_bigru_seed{seed}.pt"),
                        map_location="cpu", weights_only=False)
        net = T.Net(ck["n_features"]); net.load_state_dict(ck["state_dict"])
        P = T.predict(net, ck["mu"], ck["sd"], items)
        for d, p in zip(items, P):
            segs = T.segments(p, d["t"], ck["threshold"])
            res.setdefault(d["key"], {})[f"seed{seed}"] = [
                {"start": s, "end": e, "peak": c, "dur": e - s} for s, e, c in segs]
    T_NPZ = T.NPZ; T.NPZ = os.path.join(OUT, "npz")
    for d in items:
        r = res[d["key"]]
        for s in r["seed100"]:
            s["rotation_deg"] = yourturn_angle(d["key"], s["start"], s["end"])
        r["detect_rate"] = float((d["npose"] > 0).mean()); r["two_person_rate"] = float((d["npose"] > 1).mean())
        r["duration_s"] = float(d["t"][-1])
    T.NPZ = T_NPZ
    json.dump(res, open(os.path.join(OUT, "tulip_turns.json"), "w"), indent=1)
    for k, r in res.items():
        s = r["seed100"]
        print(k, f"det {r['detect_rate']:.2f} 2p {r['two_person_rate']:.2f}", "n_turns", [len(r[f'seed{x}']) for x in (100, 101, 102)],
              "dur", [round(z["dur"], 2) for z in s], "rot", [round(z["rotation_deg"]) for z in s])


def updrs():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from scipy.stats import spearmanr, mannwhitneyu
    from updrs_from_turns import scores, feats
    data = T.load(); tr = [d for d in data if d["split"] == "train"]
    sc = scores("train"); ytr = np.array([sc[d["key"]] for d in tr])
    Xtr = np.array([feats(d, [(L["start"], L["end"]) for L in d["lab"]])[0] for d in tr])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=3.0, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
    turns = json.load(open(os.path.join(OUT, "tulip_turns.json")))
    items = {d["key"]: d for d in load_tulip()}
    lab = pd.read_excel(os.path.join(TULIP, "Final_Label_TULIPver1.0_CVPR_2025_Mar.xlsx"), header=None)
    it = lab.iloc[2:, :5]; it.columns = ["n", "item", "c1", "c2", "c3"]
    dx = lab.iloc[3:, 6:10]; dx.columns = ["n", "d1", "d2", "d3"]
    T_NPZ = T.NPZ; T.NPZ = os.path.join(OUT, "npz")
    rows = []
    for k, r in turns.items():
        n = int(k.split("_")[1])
        g = it[(it["n"] == n) & (it["item"] == "Gait")][["c1", "c2", "c3"]].astype(float).values[0]
        dd = dx[dx["n"] == n][["d1", "d2", "d3"]].values[0].tolist()
        d = items[k]; segs = [(s["start"], s["end"]) for s in r["seed100"]]
        ev = []
        for w0 in (0, 20, 40):   # PD4T clips are 20 s: score three 20-s windows and average
            ws = [(s - w0, e - w0) for s, e in segs if w0 <= (s + e) / 2 < w0 + 20]
            m = (d["t"] >= w0) & (d["t"] < w0 + 20)
            dw = {"key": k, "t": d["t"][m] - w0, "x": d["x"][m]}
            ev.append(float(clf.predict_proba(feats(dw, ws)[0][None]) @ np.arange(len(clf.classes_))))
        durs = [s["dur"] for s in r["seed100"]]
        rows.append({"subject": k, "clin_gait": g.tolist(), "clin_mean": float(g.mean()), "dx": dd,
                     "dx_majority": max(set(dd), key=dd.count), "pred_score": float(np.mean(ev)),
                     "n_turns": len(durs), "mean_turn_dur": float(np.mean(durs)) if durs else None})
    T.NPZ = T_NPZ
    df = pd.DataFrame(rows); print(df.to_string())
    out = {"n": len(df),
           "spearman_pred_vs_clin_mean": list(map(float, spearmanr(df.pred_score, df.clin_mean))),
           "spearman_turn_dur_vs_clin_mean": list(map(float, spearmanr(df.mean_turn_dur, df.clin_mean))),
           "mae_pred_vs_clin_mean": float(np.abs(df.pred_score - df.clin_mean).mean())}
    pd_ = df[df.dx_majority == "PD"]; ht = df[df.dx_majority == "HT"]
    if len(pd_) and len(ht):
        out["turn_dur_PD_vs_HT"] = {"PD_mean": float(pd_.mean_turn_dur.mean()), "HT_mean": float(ht.mean_turn_dur.mean()),
                                    "mannwhitney_p": float(mannwhitneyu(pd_.mean_turn_dur, ht.mean_turn_dur).pvalue)}
    c = np.array(df.clin_gait.tolist())
    out["clinician_pairwise_spearman"] = [float(spearmanr(c[:, i], c[:, j])[0]) for i, j in ((0, 1), (0, 2), (1, 2))]
    print(json.dumps(out, indent=1))
    df.to_json(os.path.join(OUT, "tulip_updrs_rows.json"), orient="records", indent=1)
    json.dump(out, open(os.path.join(OUT, "tulip_updrs.json"), "w"), indent=1)


if __name__ == "__main__":
    {"extract": extract, "apply": apply, "updrs": updrs}[sys.argv[1]]()
