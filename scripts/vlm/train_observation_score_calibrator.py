"""
Train a cheap stage-2 observation->score calibrator baseline.

Uses paired observation/score examples and compares:
- heuristic bridge baseline
- learned sklearn calibrator (LOOCV when dataset is tiny)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from scripts.vlm.bridge_observation_to_score import bridge_score
except ModuleNotFoundError:
    from vlm.bridge_observation_to_score import bridge_score


DEFAULT_INPUT = Path("experiments/results/test_outputs/observation_score_pairs_v0_1.jsonl")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/observation_score_calibrator_eval_v0_1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-input", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--task", type=str, default="", help="Optional task filter for train rows")
    parser.add_argument("--eval-task", type=str, default="", help="Optional task filter for eval rows")
    parser.add_argument("--model-output", type=Path, default=None, help="Optional path to save a fitted calibrator model")
    return parser.parse_args()


def obs_to_features(task: str, observation: dict) -> dict:
    cues = observation.get("motion_cue") or []
    if isinstance(cues, str):
        cues = [cues]
    body_region = observation.get("body_region") or []
    if isinstance(body_region, str):
        body_region = [body_region]
    features = {
        "task": task,
        "answer": str(observation.get("answer", "")),
        "visibility": str(observation.get("visibility", "")),
        "uncertainty_flag": str(observation.get("uncertainty_flag", "")),
        "longitudinal_change": str(observation.get("longitudinal_change", "")),
        "cue_count": len(cues),
        "body_region_count": len(body_region),
    }
    for cue in cues:
        features[f"cue::{cue}"] = 1
    for region in body_region:
        features[f"region::{region}"] = 1
    return features


def main() -> int:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if args.task and row.get("task") != args.task:
                    continue
                rows.append(row)

    if not rows:
        raise SystemExit("No paired rows found.")

    X = [obs_to_features(row["task"], row["observation"]) for row in rows]
    y = [int(row["target_score"]) for row in rows]

    heuristic_preds = [bridge_score(row["observation"], task=row["task"])[0] for row in rows]

    # Tiny-data setup: leave-one-out style loop
    learned_preds = []
    for i in range(len(rows)):
        train_X = [x for j, x in enumerate(X) if j != i]
        train_y = [label for j, label in enumerate(y) if j != i]
        test_X = [X[i]]
        pipeline = Pipeline(
            [
                ("vec", DictVectorizer(sparse=False)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(train_X, train_y)
        pred = int(pipeline.predict(test_X)[0])
        learned_preds.append(pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    heuristic_errors = []
    learned_errors = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row, heuristic_pred, learned_pred, target in zip(rows, heuristic_preds, learned_preds, y):
            out = {
                **row,
                "heuristic_pred": heuristic_pred,
                "heuristic_abs_error": abs(target - heuristic_pred),
                "heuristic_exact": target == heuristic_pred,
                "learned_pred": learned_pred,
                "learned_abs_error": abs(target - learned_pred),
                "learned_exact": target == learned_pred,
            }
            heuristic_errors.append(out["heuristic_abs_error"])
            learned_errors.append(out["learned_abs_error"])
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "heuristic_exact_rate": sum(1 for row, pred in zip(rows, heuristic_preds) if row["target_score"] == pred) / len(rows),
        "heuristic_mae": mean(heuristic_errors),
        "learned_exact_rate": sum(1 for row, pred in zip(rows, learned_preds) if row["target_score"] == pred) / len(rows),
        "learned_mae": mean(learned_errors),
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.model_output:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("joblib is required to save the calibrator model") from exc

        fitted_pipeline = Pipeline(
            [
                ("vec", DictVectorizer(sparse=False)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        fitted_pipeline.fit(X, y)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "task": args.task or "all",
                "pipeline": fitted_pipeline,
            },
            args.model_output,
        )
        print(json.dumps({"model_output": str(args.model_output), "task": args.task or "all"}, ensure_ascii=False))

    if args.eval_input:
        eval_rows = []
        with args.eval_input.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if args.eval_task and row.get("task") != args.eval_task:
                        continue
                    eval_rows.append(row)

        pipeline = Pipeline(
            [
                ("vec", DictVectorizer(sparse=False)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)

        eval_output = args.eval_output or args.output.with_name(args.output.stem + "_eval.jsonl")
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_errors = []
        exact = 0
        with eval_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in eval_rows:
                features = obs_to_features(row["task"], row["observation"])
                pred = int(pipeline.predict([features])[0])
                heuristic_pred = bridge_score(row["observation"], task=row["task"])[0]
                target = int(row["target_score"])
                exact += int(pred == target)
                eval_errors.append(abs(target - pred))
                out = {
                    **row,
                    "heuristic_pred": heuristic_pred,
                    "learned_pred": pred,
                    "absolute_error": abs(target - pred),
                    "exact_match": pred == target,
                }
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        eval_summary = {
            "eval_input": str(args.eval_input),
            "eval_output": str(eval_output),
            "eval_rows": len(eval_rows),
            "eval_exact_rate": (exact / len(eval_rows)) if eval_rows else 0.0,
            "eval_mae": mean(eval_errors) if eval_errors else None,
        }
        print(json.dumps(eval_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
