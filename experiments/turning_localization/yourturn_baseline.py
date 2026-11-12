"""Turning-angle baseline following the "Your Turn" definition (Cheng et al., arXiv 2408.08182).

Your Turn defines a turn as pelvis rotation from initiation to completion, with >=45 deg of hip rotation
as the minimum, and measures the angle from the left-right hip and knee vectors projected on the ground
plane (average of the two). It does not segment turns automatically; here we add the simplest
segmentation consistent with that definition so it can be compared on the same localisation task:

  1. yaw(t) = mean of signed ground-plane angles of the hip and knee vectors (MediaPipe world landmarks,
     x-z plane), unwrapped and smoothed;
  2. a turn = a run of frames with |d yaw/dt| >= v_thr whose total rotation is >= 45 deg;
  3. far-end turns = turns whose mean body image height is below the clip median (person far away).

v_thr and the smoothing window are chosen on the 22 train subjects only (best F1@tIoU0.5), then the test
split is evaluated once. Also reports the rotation (deg) inside the clinician pivot labels and inside
the BiGRU predictions, using the Your Turn angle formula.
"""
import os, sys, json
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from eval_report import tiou_f1, report, model_segs, MODELS


def yaw_and_height(key):
    d = np.load(os.path.join(T.NPZ, key + ".npz"))
    t = d["t"]; Tn = len(t)
    w = T.interp(d["world"].reshape(Tn, -1)).reshape(Tn, 33, 3)
    img = T.interp(d["img"][:, :, :2].reshape(Tn, -1)).reshape(Tn, 33, 2)
    ang = []
    for a, b in ((23, 24), (25, 26)):           # hips, knees (left - right)
        v = w[:, a] - w[:, b]
        ang.append(np.unwrap(np.arctan2(v[:, 2], v[:, 0])))
    yaw = np.degrees((ang[0] + ang[1]) / 2)
    height = (img[:, 27:29, 1].mean(1) - img[:, 11:13, 1].mean(1))   # ankle - shoulder (image units)
    return t, yaw, height


def yourturn_angle(key, s, e):
    """Your Turn formula: 1/2 * sum of per-step unsigned angles of hip and knee vectors."""
    d = np.load(os.path.join(T.NPZ, key + ".npz")); t = d["t"]; Tn = len(t)
    w = T.interp(d["world"].reshape(Tn, -1)).reshape(Tn, 33, 3)
    m = (t >= s) & (t <= e)
    if m.sum() < 2:
        return 0.0
    tot = 0.0
    for a, b in ((23, 24), (25, 26)):
        v = (w[m, a] - w[m, b])[:, [0, 2]]
        cr = np.abs(v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0])
        nn = np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1) + 1e-9
        tot += np.degrees(np.arcsin(np.clip(cr / nn, 0, 1))).sum()
    return float(tot / 2)


def detect(key, v_thr, win, cache):
    if key not in cache:
        cache[key] = yaw_and_height(key)
    t, yaw, h = cache[key]
    y = uniform_filter1d(yaw, win, mode="nearest")
    v = np.abs(np.gradient(y, t))
    on = v >= v_thr; segs = []; k = 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            rot = abs(y[j] - y[k])
            if rot >= 45 and np.mean(h[k:j + 1]) < np.median(h):
                c = k + int(np.argmax(v[k:j + 1]))
                segs.append([float(t[k]), float(t[j]), float(t[c])])
            k = j + 1
        else:
            k += 1
    return segs


def main():
    data = T.load()
    tr = [d for d in data if d["split"] == "train"]; te = [d for d in data if d["split"] == "test"]
    cache = {}
    grid = {}
    for v_thr in (30, 45, 60, 90, 120, 180):
        for win in (1, 3, 5, 9):
            grid[(v_thr, win)] = tiou_f1([detect(d["key"], v_thr, win, cache) for d in tr], tr, 0.5)
    best = max(grid, key=grid.get)
    out = {"train_grid_f1_tiou0.5": {f"v{k[0]}_w{k[1]}": v for k, v in grid.items()},
           "selected": {"v_thr_deg_s": best[0], "smooth_frames": best[1]}}
    out["test_yourturn_baseline"] = report([detect(d["key"], *best, cache) for d in te], te)

    lab_ang = [yourturn_angle(d["key"], L["start"], L["end"]) for d in te for L in d["lab"]]
    segs = model_segs(os.path.join(MODELS, "turn_bigru_seed100.pt"), te)
    pred_ang = [yourturn_angle(d["key"], s[0], s[1]) for d, ss in zip(te, segs) for s in ss]
    # whole-clip rotation per clip as a reference for a full walk-turn-walk
    q = lambda a: {"median": float(np.median(a)), "iqr": [float(np.percentile(a, 25)), float(np.percentile(a, 75))], "n": len(a)}
    out["rotation_deg_inside_clinician_labels_test"] = q(lab_ang)
    out["rotation_deg_inside_bigru_predictions_test"] = q(pred_ang)
    json.dump(out, open(os.path.join(T.DATA_ROOT, "yourturn_baseline.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "train_grid_f1_tiou0.5"}, indent=1))


if __name__ == "__main__":
    main()
