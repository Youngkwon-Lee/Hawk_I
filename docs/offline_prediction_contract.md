# 오프라인 예측 결과 전달 계약 v2 — primitive 기반

작성: 2026-08-03 (v1 자유서술 → v2 온톨로지 구조로 개정) · 대상: 모델 학습 담당(영빈) → 웹 통합 담당(영권)

## v1에서 무엇이 바뀌었나

v1은 근거를 `rationale` 자유서술 문장 하나로 받았습니다. 이걸 **온톨로지 primitive 구조**로 바꿉니다.

이유는 세 가지입니다.

1. **평가할 수 있습니다.** 자유서술은 사람이 읽어야 채점되지만, primitive는 라벨과 직접 비교해 정확도·F1을 계산할 수 있습니다. C1/C3 조건의 기여를 수치로 주장할 수 있게 됩니다.
2. **화면에 그대로 씁니다.** 임상 화면의 1순위 원칙이 "점수가 아니라 관찰을 앞에"인데, 문장 한 줄로는 이걸 제대로 못 합니다. primitive는 그대로 트랙·근거 구간으로 렌더링됩니다.
3. **사람 라벨과 같은 언어를 씁니다.** 라벨링툴이 만드는 라벨과 모델 출력이 동일한 스키마가 되어, 사람 판정과 모델 판정을 같은 축에서 비교할 수 있습니다.

## 전달 형식

**JSONL** — 클립 1건당 한 줄. 스키마 정본은 `hawkeye-labeling-tool/schemas/gait_rationale_ontology_v0.schema.json` (finger tapping은 `finger_tapping_primitive_ontology_v0.schema.json`).

```json
{
  "clip_id": "PD4T_S12_gait_03",
  "task": "gait",
  "dataset": "PD4T",
  "split": "test",
  "model": "qwen3-vl-4b-c3",
  "condition": "C3",

  "quality_gate": {"status": "pass", "reasons": [], "note": ""},

  "primitives": {
    "gait_speed_reduction": {
      "observability": "observed",
      "severity": 2,
      "confidence": "medium",
      "evidence": [{"start_sec": 2.1, "end_sec": 5.4, "note": "보폭이 눈에 띄게 짧아짐"}]
    },
    "arm_swing_asymmetry": {
      "observability": "observed", "severity": 1, "confidence": "low",
      "evidence": [{"start_sec": 3.0, "end_sec": 6.2}]
    },
    "freezing_of_gait": {"observability": "unobservable", "severity": null, "confidence": "high"}
  },

  "score_anchor": {"updrs_3_10": 2, "source": "trusted_import", "confidence": "medium"},

  "reference_score": 2,
  "observed_at": "2026-08-09T10:00:00Z"
}
```

### 필수

| 필드 | 설명 |
| --- | --- |
| `clip_id`, `task`, `dataset` | v1과 동일 |
| `quality_gate.status` | `pass` 또는 `hold` |
| `primitives` | 아래 규칙 참조 |

### primitive 규칙 (라벨링툴과 동일)

- **gait 9종**: `gait_speed_reduction`, `shortened_stride`, `step_length_asymmetry`, `arm_swing_asymmetry`, `festination`, `freezing_of_gait`, `turning_impairment`, `trunk_flexion`, `postural_instability`
- 각 primitive는 `observability` (`observed` / `unobservable` / `uncertain`), `severity` (0–3 정수 또는 `null`), `confidence` (`low` / `medium` / `high`)
- **`observed`가 아니면 `severity`는 반드시 `null`** — "안 보임"을 0점(정상)으로 채우지 않습니다. 이게 온톨로지의 핵심 규칙입니다.
- **`severity > 0`이면 `evidence` 구간이 필수** — 근거 없는 양성 판정은 받지 않습니다. 화면에서 근거 구간이 영상 재생 위치로 연결되기 때문입니다.
- **`quality_gate.status`가 `hold`면 모든 severity와 score_anchor가 `null`** — 판정 불가를 0점으로 대체하지 않습니다.

### score_anchor

UPDRS 점수는 **primitive와 분리된 별도 앵커**입니다. primitive severity를 합산해 만들지 마세요. 모델이 점수를 직접 예측했다면 `source`를 `trusted_import`로 두고, 예측하지 않았다면 `not_scored`로 두면 됩니다.

### 선택

`model`, `condition`(C0~C3), `reference_score`(정답 라벨), `subject_ref`, `observed_at`, `metrics`(kinematic 수치)

## 적재 방법 (영권 실행)

```bash
# 미리보기 — 아무것도 쓰지 않음
python backend/scripts/ingest_model_predictions.py preds.jsonl \
    --subject-person-id <demo-person-uuid>

# 실제 적재
python backend/scripts/ingest_model_predictions.py preds.jsonl \
    --subject-person-id <demo-person-uuid> --apply
```

v1 형식(`predicted_score` + `rationale`)도 계속 받습니다. primitive가 없으면 점수만 표시되고, 있으면 화면에 primitive 트랙이 뜹니다.

## 데이터 경계

모든 행에 `research_provenance`(연구 예측 표시, dataset, split, 모델, 조건)가 붙고 `category`에 `research-prediction`이 추가됩니다. 연구 데이터셋 예측이 임상 기록으로 오인되지 않게 하기 위한 것입니다. 원본 영상·라벨 DB·개인식별정보는 이 경로로 전달하지 않습니다.
