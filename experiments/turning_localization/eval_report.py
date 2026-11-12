"""Evaluation report for far-end turn localisation on the PD4T official test split.

Primary metric: segment F1 at temporal IoU (tIoU) 0.5, plus F1 at 0.3/0.7 and the mean over
tIoU 0.1:0.1:0.7. The centre-match F1 (label centre inside a predicted segment +/-0.5 s) is kept as a
secondary, lenient metric: predicting the whole clip as one segment already scores ~0.77 on it.

Also reports two sanity controls:
  - trivial baselines (one fixed segment at the mean train turn time; the whole clip as one segment)
  - a label-shuffle control: same model trained with each train clip given another clip's labels
and a subject-level bootstrap 95% CI for the trained model.

Needs the private data described in config.py and the checkpoints in ../models.
"""
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
import rule_height, rule_path

MODELS = os.environ.get("PD4T_TURN_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
TIOUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SEEDS = (100, 101, 102)


def tiou_f1(segs_per_clip, items, thr):
    hit = n_lab = n_pred = 0
    for segs, d in zip(segs_per_clip, items):
        n_lab += len(d["lab"]); n_pred += len(segs)
        # greedy one-to-one matching by best tIoU
        pairs = sorted(((T.tiou((L["start"], L["end"]), s), i, j)
                        for i, L in enumerate(d["lab"]) for j, s in enumerate(segs)), reverse=True)
        ul, us = set(), set()
        for v, i, j in pairs:
            if v >= thr and i not in ul and j not in us:
                ul.add(i); us.add(j); hit += 1
    return 2 * hit / max(n_lab + n_pred, 1)


def report(segs, items):
    r = {f"f1_tiou{t}": tiou_f1(segs, items, t) for t in (0.3, 0.5, 0.7)}
    r["f1_tiou_mean_0.1_0.7"] = float(np.mean([tiou_f1(segs, items, t) for t in TIOUS]))
    c = T.evaluate(segs, items)
    r.update({"centre_f1_lenient": c["f1"], "recall_centre": c["recall"], "precision_centre": c["precision"],
              "centre_err_med_s": c["centre_err_med"], "n_lab": c["n_lab"], "n_pred": c["n_pred"]})
    return r


def model_segs(ckpt_path, items):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = T.Net(ck["n_features"]); net.load_state_dict(ck["state_dict"])
    P = T.predict(net, ck["mu"], ck["sd"], items)
    return [T.segments(p, d["t"], ck["threshold"]) for p, d in zip(P, items)]


def rule_segs(mod, items):
    out = []
    for d in items:
        pred, _ = mod.detect(os.path.join(T.NPZ, d["key"] + ".npz"))
        out.append([[p["start"], p["end"], p["center"]] for p in pred if p["kind"] == "far"])
    return out


def main():
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    out = {"checks": {
        "subject_overlap": sorted({d["subj"] for d in tr} & {d["subj"] for d in te}),
        "visit_overlap": len({d["key"] for d in tr} & {d["key"] for d in te}),
        "n_subjects": [len({d["subj"] for d in tr}), len({d["subj"] for d in te})],
        "n_clips": [len(tr), len(te)]}}
    rows = {}
    seg_by_seed = {}
    for s in SEEDS:
        seg_by_seed[s] = model_segs(os.path.join(MODELS, f"turn_bigru_seed{s}.pt"), te)
        rows[f"BiGRU seed {s}"] = report(seg_by_seed[s], te)
    keys = list(rows[f"BiGRU seed {SEEDS[0]}"].keys())
    rows["BiGRU mean"] = {k: float(np.mean([rows[f"BiGRU seed {s}"][k] for s in SEEDS])) for k in keys
                          if rows[f"BiGRU seed {SEEDS[0]}"][k] is not None}
    rows["BiGRU sd"] = {k: float(np.std([rows[f"BiGRU seed {s}"][k] for s in SEEDS])) for k in keys
                        if rows[f"BiGRU seed {SEEDS[0]}"][k] is not None}
    rows["Rule: body-height extrema"] = report(rule_segs(rule_height, te), te)
    rows["Rule: walking-path reversal"] = report(rule_segs(rule_path, te), te)

    cs = [(L["start"] + L["end"]) / 2 for d in tr for L in d["lab"]]
    ds = [L["end"] - L["start"] for d in tr for L in d["lab"]]
    c, w = float(np.mean(cs)), float(np.mean(ds))
    rows["Trivial: fixed segment"] = report([[[c - w / 2, c + w / 2, c]] for _ in te], te)
    rows["Trivial: whole clip"] = report([[[float(d["t"][0]), float(d["t"][-1]), float(d["t"][len(d["t"]) // 2])]]
                                          for d in te], te)

    rng = random.Random(7); perm = list(range(len(tr))); rng.shuffle(perm)
    sh = []
    for i, d in enumerate(tr):
        y = np.zeros(len(d["t"]), np.float32)
        for L in tr[perm[i]]["lab"]:
            y[(d["t"] >= L["start"]) & (d["t"] <= L["end"])] = 1
        sh.append({**d, "y": y})
    net, mu, sd, _ = T.train(sh, 40, seed=100)
    P = T.predict(net, mu, sd, te)
    rows["Control: shuffled-label BiGRU"] = report([T.segments(p, d["t"], 0.6) for p, d in zip(P, te)], te)
    out["rows"] = rows

    subs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0)
    seg_of = {d["key"]: s for s, d in zip(seg_by_seed[SEEDS[0]], te)}
    b5, bc = [], []
    for _ in range(1000):
        items = [d for s in rs.choice(subs, len(subs), replace=True) for d in te if d["subj"] == s]
        segs = [seg_of[d["key"]] for d in items]
        b5.append(tiou_f1(segs, items, 0.5)); bc.append(T.evaluate(segs, items)["f1"])
    out["bootstrap_subject_95ci_seed%d" % SEEDS[0]] = {
        "f1_tiou0.5": [float(np.percentile(b5, 2.5)), float(np.percentile(b5, 97.5))],
        "centre_f1_lenient": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]}
    out["per_subject_f1_tiou0.5_seed%d" % SEEDS[0]] = {
        s: round(tiou_f1([seg_of[d["key"]] for d in te if d["subj"] == s], [d for d in te if d["subj"] == s], 0.5), 3)
        for s in subs}
    dst = os.path.join(T.DATA_ROOT, "eval_report.json")
    json.dump(out, open(dst, "w"), indent=1)
    for name, r in rows.items():
        print(f"{name:32s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
