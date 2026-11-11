"""Rule-based turn candidates from monocular pose; compare with clinician far-end turn labels.

Signals (image coordinates, camera fixed at the walkway end):
- h(t): body height in image (shoulder-mid to ankle-mid). Walking away -> h shrinks,
  walking back -> h grows. Far-end turn ~ local minimum, near-end turn ~ local maximum.
- f(t): facing sign from left/right shoulder order in the image
  (subject faces camera -> subject's left shoulder appears on image right).
Turn interval: around each extremum, the span where |dh/dt| is small, extended to
cover the facing flip.
"""
import os, glob, json, os, csv
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from config import DATA_ROOT, LABEL_DIR, OFFICIAL_DIR
NPZ = os.path.join(DATA_ROOT, "npz")
LABELS = [os.path.join(LABEL_DIR, f"{s}.jsonl") for s in ("train", "validation", "test")]
OFFICIAL = OFFICIAL_DIR
L_SH, R_SH, L_AN, R_AN, L_HIP, R_HIP = 11, 12, 27, 28, 23, 24


def interp_nan(x):
    x = x.copy(); n = np.isnan(x)
    if n.all():
        return x
    x[n] = np.interp(np.flatnonzero(n), np.flatnonzero(~n), x[~n])
    return x


def detect(npz_path, min_gap=2.0):
    d = np.load(npz_path)
    t, img = d["t"], d["img"]
    if len(t) < 30:
        return [], {}
    dt = float(np.median(np.diff(t)))
    sh_y = interp_nan((img[:, L_SH, 1] + img[:, R_SH, 1]) / 2)
    an_y = interp_nan((img[:, L_AN, 1] + img[:, R_AN, 1]) / 2)
    h = an_y - sh_y
    win = max(5, int(round(1.0 / dt)) | 1)
    hs = savgol_filter(h, win, 2)
    v = np.gradient(hs, dt)
    face = interp_nan(img[:, L_SH, 0] - img[:, R_SH, 0])  # >0: facing camera
    fs = savgol_filter(face, win, 2)
    prom = 0.45 * (np.nanpercentile(hs, 95) - np.nanpercentile(hs, 5) + 1e-6)
    dist = max(1, int(min_gap / dt))
    turns = []
    for kind, sig in (("far", -hs), ("near", hs)):
        peaks, _ = find_peaks(sig, prominence=prom, distance=dist)
        for p in peaks:
            if p < 3 or p > len(t) - 4:  # ignore clip edges: start/stop, not a turn
                continue
            vthr = 0.35 * np.nanmax(np.abs(v))
            a = p
            while a > 0 and abs(v[a]) < vthr:
                a -= 1
            b = p
            while b < len(t) - 1 and abs(v[b]) < vthr:
                b += 1
            # extend to the facing flip if it lies just outside
            sgn = np.sign(fs)
            flips = [i for i in range(max(1, a - int(1 / dt)), min(len(t), b + int(1 / dt))) if sgn[i] != sgn[i - 1]]
            if flips:
                a, b = min(a, flips[0]), max(b, flips[-1])
            turns.append({"kind": kind, "start": float(t[a]), "end": float(t[b]), "center": float(t[p])})
    turns.sort(key=lambda x: x["start"])
    qc = {"detect_rate": float((d["npose"] > 0).mean()), "n_frames": int(len(t))}
    return turns, qc


def official_split():
    m = {}
    for s in ("train", "test"):
        for r in csv.reader(open(f"{OFFICIAL}/{s}.csv")):
            v, p = r[0].rsplit("_", 1)
            m[(v, p)] = s
    return m


def main():
    split = official_split()
    labels = {}
    for f in LABELS:
        for line in open(f):
            r = json.loads(line)
            labels[f"{r['clip_id']}_{r['patient_id'].zfill(3)}"] = r
    rows, hits, nlab, nfar, center_err, near_total = [], 0, 0, 0, [], 0
    per_split = {}
    for path in sorted(glob.glob(f"{NPZ}/*.npz")):
        key = os.path.basename(path)[:-4]
        r = labels.get(key)
        if r is None:
            continue
        pred, qc = detect(path)
        far = [p for p in pred if p["kind"] == "far"]
        near = [p for p in pred if p["kind"] == "near"]
        near_total += len(near)
        lab = r.get("turning_intervals") or []
        sp = split.get((r["clip_id"], r["patient_id"].zfill(3)), "?")
        st = per_split.setdefault(sp, {"labels": 0, "hit": 0, "far_pred": 0, "near_pred": 0, "clips": 0})
        st["clips"] += 1; st["labels"] += len(lab); st["far_pred"] += len(far); st["near_pred"] += len(near)
        used = set()
        for L in lab:
            c = (L["start"] + L["end"]) / 2
            best = None
            for j, p in enumerate(far):
                if j in used:
                    continue
                # hit: label centre within the predicted far-turn interval +-0.5 s
                if p["start"] - 0.5 <= c <= p["end"] + 0.5:
                    e = abs(p["center"] - c)
                    if best is None or e < best[1]:
                        best = (j, e)
            if best:
                used.add(best[0]); hits += 1; st["hit"] += 1; center_err.append(best[1])
        nlab += len(lab); nfar += len(far)
        rows.append({"clip": key, "split": sp, "score": r.get("updrs_3_10"), "labels": lab,
                     "pred": pred, **qc})
    json.dump(rows, open(os.path.join(DATA_ROOT, "turn_candidates.json"), "w"))
    print(f"clips {len(rows)} | labelled far turns {nlab} | predicted far {nfar} near {near_total}")
    print(f"label recall (far) {hits}/{nlab} = {hits/max(nlab,1):.3f} | far precision vs labels {hits}/{nfar} = {hits/max(nfar,1):.3f}")
    if center_err:
        print(f"centre error median {np.median(center_err):.2f}s, p90 {np.percentile(center_err,90):.2f}s")
    for k, v in per_split.items():
        print(k, v, f"recall {v['hit']/max(v['labels'],1):.3f}")


if __name__ == "__main__":
    main()
