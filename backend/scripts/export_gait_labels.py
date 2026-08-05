#!/usr/bin/env python3
"""Export labeled gait clips as train/validation/test JSONL for model training.

    python backend/scripts/export_gait_labels.py \\
        --db ~/rehab_labeling_server/review_apps/exports/rehab_labeling.sqlite3 \\
        --out ~/gait_export

Writes train.jsonl / validation.jsonl / test.jsonl, a summary.json, and a
README.md describing the fields and their known gaps. Refuses to write if a
patient appears in more than one split, since that would invalidate the
evaluation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.label_export import (  # noqa: E402
    SPLIT_BY_PROJECT,
    TRAINABLE_FIELDS,
    patient_overlap,
    summarise,
    to_training_record,
)
from services.primitive_eval import GAIT_LABEL_TO_PRIMITIVE  # noqa: E402

README = """# PD4T gait labels — training export

생성: {generated} · 라벨러: 단일 (dongsung) · 총 {total}클립

## 파일

| 파일 | 클립 | 환자 |
| --- | --- | --- |
{split_rows}

환자는 split 간 겹치지 않습니다(자동 검사 통과). 한 명이라도 겹치면 이 파일은 생성되지 않습니다.

## 레코드 형식

```json
{sample}
```

## 중증도 값

라벨링 화면은 텍스트로 저장합니다. 여기서는 숫자로 변환했습니다.

| 원본 | 변환 |
| --- | --- |
| 정상 / none / normal | 0 |
| 경미↓ / mild | 1 |
| 중등↓ / moderate | 2 |
| 심함↓ / severe | 3 |

**`null`은 0이 아닙니다.** 라벨러가 비워둔 값이며, 온톨로지 규칙상 "관찰되지 않음"입니다.
0(정상)으로 채우면 가려진 소견을 건강한 것으로 학습시키게 됩니다. 손실 계산에서 제외하세요.

## 알려진 한계

1. **`turning_impairment`는 학습 필드에서 제외했습니다.** 학습셋의 99.6%가 비어 있어 신호가 없습니다.
   다만 **회전 구간(`turning_intervals`)은 96%가 채워져 있어** 그대로 넣었습니다. 시점 예측 과제에 쓸 수 있습니다.
2. **`freezing_of_gait` 중증도도 대부분 비어 있습니다.** 대신 `freezing_intervals`가 신뢰할 수 있는 신호이고,
   양성은 전체 28클립뿐입니다. 학습보다 평가/벤치마크에 적합합니다.
3. **test 세트에 3점이 없습니다.** 3점 예측 성능은 이 split으로 측정할 수 없습니다.
4. **클래스 불균형** — 학습셋의 53%가 0점입니다. "항상 0"이 53% 정확도를 냅니다.
   정확도를 보고할 때 이 기준선을 반드시 함께 제시하세요.
5. **라벨러가 1명**이라 라벨러 간 일치도를 계산할 수 없습니다. 논문에는 "단일 숙련 라벨러 기준"으로 명시하세요.

## 사용 가능한 primitive {n_primitives}개

{primitive_list}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--item", default="gait")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"error: {args.db} not found", file=sys.stderr)
        return 2

    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    records = []
    for project, split in SPLIT_BY_PROJECT.items():
        rows = connection.execute(
            "SELECT raw_json FROM label_latest WHERE project_id=? AND updrs_item=?",
            (project, args.item),
        )
        for (raw,) in rows:
            try:
                label = json.loads(raw)
            except json.JSONDecodeError:
                continue
            records.append(to_training_record(label, split))

    if not records:
        print("error: no labels found", file=sys.stderr)
        return 1

    overlaps = patient_overlap(records)
    if overlaps:
        print("error: the same patient appears in more than one split:", file=sys.stderr)
        for pair, patients in overlaps.items():
            print(f"  {pair}: {', '.join(patients)}", file=sys.stderr)
        print("Refusing to write - this would invalidate the evaluation.", file=sys.stderr)
        return 1

    summary = summarise(records)
    args.out.mkdir(parents=True, exist_ok=True)

    for split in sorted(summary):
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {path.name}: {summary[split]['clips']}클립, 환자 {summary[split]['patients']}명")

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from datetime import datetime, timezone
    sample = dict(records[0])
    sample["media_path"] = "<clip path>"
    split_rows = "\n".join(
        f"| {split}.jsonl | {summary[split]['clips']} | {summary[split]['patients']} |"
        for split in sorted(summary)
    )
    (args.out / "README.md").write_text(
        README.format(
            generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            total=len(records),
            split_rows=split_rows,
            sample=json.dumps(sample, ensure_ascii=False, indent=2),
            n_primitives=len(TRAINABLE_FIELDS),
            primitive_list="\n".join(f"- `{GAIT_LABEL_TO_PRIMITIVE[f]}`" for f in TRAINABLE_FIELDS),
        ),
        encoding="utf-8",
    )

    print(f"\n환자 누수 검사: 통과 (split 간 중복 0명)")
    print(f"요약: {args.out / 'summary.json'} · 설명: {args.out / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
