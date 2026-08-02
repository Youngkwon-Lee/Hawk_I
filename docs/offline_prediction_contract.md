# 오프라인 예측 결과 전달 계약 (KHF 데모용)

작성: 2026-08-02 · 대상: 모델 학습 담당(영빈) → 웹 통합 담당(영권)

## 왜 이 형식인가

KHF 발표에서는 모델을 웹에 **실시간 서빙하지 않습니다.** 백엔드가 도는 집 데스크탑에는 GPU가 없고(RAM 11GB / CPU 6코어), 8/12 제출 마감까지 서빙 인프라를 뚫고 검증할 여유가 없습니다.

대신 **학습용 GPU에서 테스트셋을 미리 추론**하고, 그 결과를 공통 환자 타임라인(`observations`)에 적재해 웹사이트가 표시합니다. 화면에는 실제 모델 출력(점수 + rationale)이 그대로 뜨고, 추가 서빙 인프라는 필요 없습니다.

## 전달 형식

**JSONL** — 예측 1건당 한 줄.

```json
{"clip_id": "PD4T_S12_gait_03", "task": "gait", "predicted_score": 2, "rationale": "보폭 감소와 팔 흔들림 저하가 관찰됨", "dataset": "PD4T", "split": "test", "model": "qwen3-vl-4b-c3", "condition": "C3", "confidence": 0.81, "true_score": 2, "subject_ref": "S12", "observed_at": "2026-08-09T10:00:00Z"}
```

### 필수 필드

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `clip_id` | string | 클립 고유 id |
| `task` | string | `gait`, `finger_tapping`, `hand_movement`, `pronation_supination` 중 하나. 그 외 값도 받지만 UPDRS 항목 코드가 자동 매핑되지 않습니다 |
| `predicted_score` | number | 0~4 (범위 밖이면 해당 줄만 거부되고 나머지는 적재됩니다) |
| `dataset` | string | 예: `PD4T`. 연구 데이터임을 표시하기 위해 필수입니다 |

### 선택 필드 (있으면 화면이 풍부해짐)

| 필드 | 설명 |
| --- | --- |
| `rationale` | 모델이 생성한 근거 문장. 발표에서 XAI를 보여주는 핵심이라 **가능하면 꼭 넣어주세요** |
| `condition` | `C0`~`C3` 실험 조건. 조건별 비교 표시에 사용 |
| `model` | 모델/체크포인트 식별자 (예: `qwen3-vl-4b-c3`) |
| `confidence` | 모델 신뢰도 |
| `true_score` | 정답 라벨. 예측 대비 비교 표시에 사용 |
| `subject_ref` | 연구 피험자 식별자 (예: `S12`) |
| `observed_at` | ISO8601. 없으면 적재 시각 사용 |

## 주의: 이름 짓기 규칙

`model` + `condition` + `clip_id` 조합으로 고유 id가 만들어집니다. 따라서:

- **같은 파일을 다시 적재하면 덮어쓰기**가 되어 중복이 생기지 않습니다. 결과를 수정해 다시 보내도 안전합니다.
- **조건이 다르면 다른 행**이 됩니다. C0~C3를 모두 보내면 4개 행이 각각 남습니다.
- 반대로 `model`이나 `condition`을 비워두고 여러 실험 결과를 보내면 서로 덮어씁니다. 여러 실험을 구분해 보여주려면 두 필드를 꼭 채워주세요.

## 적재 방법 (영권 실행)

```bash
# 미리보기 — 아무것도 쓰지 않음
python backend/scripts/ingest_model_predictions.py preds.jsonl \
    --subject-person-id <demo-person-uuid>

# 실제 적재
python backend/scripts/ingest_model_predictions.py preds.jsonl \
    --subject-person-id <demo-person-uuid> --apply
```

기본이 dry-run이라 먼저 내용을 확인한 뒤 `--apply`로 씁니다. 형식이 깨진 줄은 이유와 함께 보고되고 건너뛰며, 나머지는 정상 적재됩니다.

## 데이터 경계

적재되는 모든 행에는 `research_provenance` 블록이 붙습니다:

```json
{"is_research_prediction": true, "is_clinical_record": false,
 "serving_mode": "offline_batch", "dataset": "PD4T", "split": "test",
 "research_subject_ref": "S12", "model": "...", "condition": "C3",
 "reference_score": 2}
```

연구 데이터셋 예측이 임상 기록으로 오인되지 않도록 하기 위한 표시이며, `category`에도 `research-prediction`이 추가됩니다. **연구 피험자를 실제 환자로 표시하지 않습니다** — 데모 타임라인은 명시적으로 지정한 데모 person에 귀속됩니다.

원본 영상, 라벨 DB, 개인식별정보는 이 경로로 전달하지 않습니다. JSONL에는 위 표의 필드만 담아주세요.
