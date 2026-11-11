"""Score VLM temporal-grounding predictions with the same evaluator as eval_report.py.

usage: python eval_vlm.py predictions.jsonl
Each line: {"key": "<visit>_<pid>", "pred": [[start, end, centre], ...], "parsed": bool}
Clips missing from the file count as empty predictions.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_sup as T
from eval_report import report, tiou_f1


def main(path):
    preds = {}
    n_unparsed = 0
    for line in open(path):
        r = json.loads(line)
        preds[r["key"]] = r["pred"]
        n_unparsed += 0 if r.get("parsed", True) else 1
    te = [d for d in T.load() if d["split"] == "test"]
    segs = [preds.get(d["key"], []) for d in te]
    out = {"file": os.path.basename(path), "n_clips_scored": len(te),
           "n_missing": sum(d["key"] not in preds for d in te), "n_unparsed": n_unparsed,
           "test": report(segs, te)}
    subs = sorted({d["subj"] for d in te}); rs = np.random.default_rng(0); b = []
    seg_of = dict(zip([d["key"] for d in te], segs))
    for _ in range(1000):
        items = [d for s in rs.choice(subs, len(subs), replace=True) for d in te if d["subj"] == s]
        b.append(tiou_f1([seg_of[d["key"]] for d in items], items, 0.5))
    out["bootstrap_subject_95ci_f1_tiou0.5"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    errs = [abs(p[0][2] - (d["lab"][0]["start"] + d["lab"][0]["end"]) / 2)
            for d, p in zip(te, segs) if p and d["lab"]]
    out["first_pred_vs_first_label_centre_abs_err_s_median"] = float(np.median(errs)) if errs else None
    print(json.dumps(out, indent=1))
    json.dump(out, open(path + ".eval.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
