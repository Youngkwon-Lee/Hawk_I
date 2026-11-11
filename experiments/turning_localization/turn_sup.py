"""Supervised far-end turn localisation from MediaPipe pose (PD4T gait, official split).

Labels: clinician far-end turnaround pivot intervals. Frame target = 1 inside a label.
Model selection: 5-fold subject-grouped CV inside the 22 official train subjects.
Test (8 official test subjects) is evaluated once with the selected setting (3 seeds).
"""
import os, glob, json, os, csv, sys, random
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(4)
from config import DATA_ROOT, LABEL_DIR, OFFICIAL_DIR
NPZ = os.path.join(DATA_ROOT, "npz")
OUTDIR = os.path.join(DATA_ROOT, "sup")
os.makedirs(OUTDIR, exist_ok=True)
PTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def interp(a):
    a = a.copy()
    for j in range(a.shape[1]):
        x = a[:, j]; n = np.isnan(x)
        if n.all():
            x[:] = 0
        elif n.any():
            x[n] = np.interp(np.flatnonzero(n), np.flatnonzero(~n), x[~n])
    return a


def features(d):
    img, world, t = d["img"], d["world"], d["t"]
    T = len(t)
    xy = img[:, :, :2].reshape(T, -1)
    xy = interp(xy).reshape(T, 33, 2)
    vis = interp(img[:, :, 3])
    hip = (xy[:, 23] + xy[:, 24]) / 2
    sh = (xy[:, 11] + xy[:, 12]) / 2
    torso = np.linalg.norm(sh - hip, axis=1, keepdims=True) + 1e-3
    rel = (xy[:, PTS] - hip[:, None]) / torso[:, :, None]
    an = (xy[:, 27] + xy[:, 28]) / 2
    h = np.log(np.clip(an[:, 1] - sh[:, 1], 1e-3, None))
    face = (xy[:, 11, 0] - xy[:, 12, 0]) / torso[:, 0]
    w = interp(world.reshape(T, -1)).reshape(T, 33, 3)
    sv = w[:, 11] - w[:, 12]
    yaw = np.arctan2(sv[:, 2], sv[:, 0])
    base = np.concatenate([rel.reshape(T, -1), vis[:, [0, 7, 8]], h[:, None], hip[:, :1], face[:, None],
                           np.sin(yaw)[:, None], np.cos(yaw)[:, None]], 1)
    vel = np.gradient(base, axis=0)
    f = np.concatenate([base, vel], 1).astype(np.float32)
    return np.nan_to_num(f), t


def load():
    labels = {}
    for s in ("train", "validation", "test"):
        for line in open(os.path.join(LABEL_DIR, f"{s}.jsonl")):
            r = json.loads(line)
            labels[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    split = {}
    for s in ("train", "test"):
        for r in csv.reader(open(os.path.join(OFFICIAL_DIR, f"{s}.csv"))):
            v, p = r[0].rsplit("_", 1)
            split[f"{v}_{p}"] = s
    data = []
    for path in sorted(glob.glob(f"{NPZ}/*.npz")):
        key = os.path.basename(path)[:-4]
        if key not in labels or key not in split:
            continue
        f, t = features(np.load(path))
        lab = labels[key].get("turning_intervals") or []
        y = np.zeros(len(t), np.float32)
        for L in lab:
            y[(t >= L["start"]) & (t <= L["end"])] = 1
        data.append({"key": key, "subj": key.rsplit("_", 1)[1], "split": split[key], "x": f, "y": y, "t": t, "lab": lab})
    return data


class Net(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(F, 64, 5, padding=2), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h).squeeze(-1)


def batchify(items, mu, sd):
    T = max(len(i["y"]) for i in items)
    F = items[0]["x"].shape[1]
    X = np.zeros((len(items), T, F), np.float32); Y = np.zeros((len(items), T), np.float32); M = np.zeros((len(items), T), np.float32)
    for k, i in enumerate(items):
        n = len(i["y"]); X[k, :n] = (i["x"] - mu) / sd; Y[k, :n] = i["y"]; M[k, :n] = 1
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


def train(items, epochs, seed, eval_at=()):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    allx = np.concatenate([i["x"] for i in items]); mu, sd = allx.mean(0), allx.std(0) + 1e-6
    net = Net(allx.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    pos = np.concatenate([i["y"] for i in items]).mean()
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3)))
    snaps = {}
    for ep in range(1, epochs + 1):
        net.train(); random.shuffle(items)
        for b in range(0, len(items), 16):
            X, Y, M = batchify(items[b:b + 16], mu, sd)
            loss = (lossf(net(X), Y) * M).sum() / M.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in eval_at:
            snaps[ep] = ({k: v.clone() for k, v in net.state_dict().items()})
    return net, mu, sd, snaps


def predict(net, mu, sd, items):
    net.eval(); out = []
    with torch.no_grad():
        for i in items:
            X, _, _ = batchify([i], mu, sd)
            out.append(torch.sigmoid(net(X))[0].numpy())
    return out


def segments(p, t, thr, min_len=2, gap=0.3):
    on = p >= thr; segs = []; k = 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j - k + 1 >= min_len:
                segs.append([float(t[k]), float(t[j]), float(t[k + int(np.argmax(p[k:j + 1]))])])
            k = j + 1
        else:
            k += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def tiou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); uni = max(a[1], b[1]) - min(a[0], b[0])
    return inter / uni if uni > 0 else 0


def evaluate(preds_per_clip, items):
    """preds_per_clip: list of [start,end,center]; returns centre-match and tIoU metrics."""
    m = {"lab": 0, "pred": 0, "hit": 0, "err": [], "t3": 0, "t5": 0}
    for segs, i in zip(preds_per_clip, items):
        lab = i["lab"]; m["lab"] += len(lab); m["pred"] += len(segs)
        used = set()
        for L in lab:
            c = (L["start"] + L["end"]) / 2; best = None
            for j, s in enumerate(segs):
                if j not in used and s[0] - 0.5 <= c <= s[1] + 0.5:
                    e = abs(s[2] - c)
                    if best is None or e < best[1]:
                        best = (j, e)
            if best:
                used.add(best[0]); m["hit"] += 1; m["err"].append(best[1])
        for thr, key in ((0.3, "t3"), (0.5, "t5")):
            used = set()
            for L in lab:
                for j, s in enumerate(segs):
                    if j not in used and tiou((L["start"], L["end"]), s) >= thr:
                        used.add(j); m[key] += 1; break
    R = m["hit"] / max(m["lab"], 1); P = m["hit"] / max(m["pred"], 1)
    f = lambda h: 2 * h / max(m["lab"] + m["pred"], 1)
    return {"recall": R, "precision": P, "f1": 2 * P * R / max(P + R, 1e-9),
            "f1_tiou0.3": f(m["t3"]), "f1_tiou0.5": f(m["t5"]),
            "centre_err_med": float(np.median(m["err"])) if m["err"] else None, "n_lab": m["lab"], "n_pred": m["pred"]}


def main():
    data = load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    subs = sorted({d["subj"] for d in tr}); rng = random.Random(0); rng.shuffle(subs)
    folds = [subs[k::5] for k in range(5)]
    EPOCHS = (20, 40); THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
    cv = {(e, th): [] for e in EPOCHS for th in THRS}
    for k, fs in enumerate(folds):
        trn = [d for d in tr if d["subj"] not in fs]; val = [d for d in tr if d["subj"] in fs]
        net, mu, sd, snaps = train(list(trn), max(EPOCHS), seed=k, eval_at=EPOCHS)
        for e in EPOCHS:
            net.load_state_dict(snaps[e]); P = predict(net, mu, sd, val)
            for th in THRS:
                segs = [segments(p, d["t"], th) for p, d in zip(P, val)]
                cv[(e, th)].append(evaluate(segs, val)["f1"])
        print(f"fold {k} done", flush=True)
    summary = {f"ep{e}_thr{th}": float(np.mean(v)) for (e, th), v in cv.items()}
    best = max(cv, key=lambda k: np.mean(cv[k]))
    print("CV mean F1 (centre match):", json.dumps(summary, indent=0))
    print("selected", best, flush=True)
    e, th = best
    res = []
    for seed in (0, 1, 2):
        net, mu, sd, _ = train(list(tr), e, seed=100 + seed)
        P = predict(net, mu, sd, te)
        segs = [segments(p, d["t"], th) for p, d in zip(P, te)]
        r = evaluate(segs, te); res.append(r); print("test seed", seed, json.dumps(r), flush=True)
        if seed == 0:
            json.dump([{"key": d["key"], "pred": s, "lab": d["lab"]} for s, d in zip(segs, te)], open(f"{OUTDIR}/test_pred_seed0.json", "w"))
    keys = ["recall", "precision", "f1", "f1_tiou0.3", "f1_tiou0.5", "centre_err_med"]
    agg = {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))) for k in keys}
    print("TEST mean±sd over 3 seeds:", json.dumps(agg))
    json.dump({"cv": summary, "selected": {"epochs": e, "thr": th}, "test": res, "test_agg": agg}, open(f"{OUTDIR}/results.json", "w"), indent=1)

    # rule-based v1 / v2 on the same test clips with the same evaluator
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("rule_height", "rule_path"):
        m = __import__(mod)
        segs = []
        for d in te:
            pred, _ = m.detect(f"{NPZ}/{d['key']}.npz")
            segs.append([[p["start"], p["end"], p["center"]] for p in pred if p["kind"] == "far"])
        print(f"RULE {mod} test:", json.dumps(evaluate(segs, te)))


if __name__ == "__main__":
    main()
