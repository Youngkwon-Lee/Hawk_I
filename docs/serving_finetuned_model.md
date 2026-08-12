# 파인튜닝 모델을 웹사이트에 연결하기

작성: 2026-08-07 · 대상: 모델 학습 담당(영빈), 플랫폼 담당(영권)

## 지금 상태

**웹 쪽 연결은 끝났습니다.** 환경변수 두 개만 설정하면 업로드된 영상이 파인튜닝 모델로 분석되고, 결과가 임상 화면에 그대로 뜹니다.

```bash
HAWKEYE_VLM_BASE_URL=https://<엔드포인트>/v1
HAWKEYE_VLM_MODEL=<모델 이름>
HAWKEYE_VLM_API_KEY=<필요한 경우만>
```

설정이 없으면 **아무 일도 일어나지 않습니다** — 기존 분석이 그대로 동작합니다. 엔드포인트가 죽어 있어도 마찬가지로 기존 결과를 냅니다. 데모 중에 GPU가 꺼져도 화면이 깨지지 않습니다.

## 모델이 지켜야 할 출력 형식

OpenAI 호환 `/chat/completions`로 프레임을 받고, **JSON만** 반환하면 됩니다.

```json
{
  "primitives": {
    "gait_speed_reduction": {"observability": "observed", "severity": 2, "confidence": "medium"},
    "freezing_of_gait": {"observability": "unobservable", "severity": null, "confidence": "high"}
  },
  "turning_intervals": [{"start": 8.7, "end": 9.4}],
  "freezing_intervals": [],
  "updrs_3_10": 2,
  "summary": "보폭 감소와 팔 흔들림 저하가 관찰됨"
}
```

- `primitives` 8개: `gait_speed_reduction`, `shortened_stride`, `step_length_asymmetry`, `arm_swing_asymmetry`, `festination`, `freezing_of_gait`, `trunk_flexion`, `postural_instability`
- `observability`: `observed` / `unobservable` / `uncertain`
- `severity`: 0~3, **`observed`가 아니면 반드시 `null`**

프롬프트는 서버가 자동으로 만들어 보냅니다 — **영상 길이와 프레임 수를 계산해 넣어주므로** 모델이 초 단위로 답할 수 있습니다. 동결 제외 기준(의도적 정지는 동결이 아님)도 프롬프트에 포함돼 있습니다.

## 서버가 알아서 걸러내는 것

모델이 형식을 어겨도 화면이 깨지지 않게 처리합니다.

| 상황 | 처리 |
| --- | --- |
| 코드펜스·잡담이 섞임 | JSON만 추출 |
| 없는 primitive 이름 | 무시 |
| severity가 0~3 밖 | 해당 항목 버림 |
| **관찰 안 됐는데 severity를 씀** | **`null`로 강제** — 안 보인 걸 정상으로 만들지 않음 |
| 구간이 영상 길이를 넘음 | 잘라내거나 버림 |
| JSON이 아예 아님 | 기존 분석 결과 유지 |

## GPU는 어디에 띄우나

**현재 백엔드 서버(집 데스크탑)에는 GPU가 없습니다.** CPU 6코어 / RAM 11GB라 4B 모델도 못 돌립니다. 별도 GPU 엔드포인트가 필요합니다.

- **RunPod** — 저장소에 이미 `scripts/runpod/Dockerfile.qwen-vl-server`와 기동 스크립트가 있습니다. OpenAI 호환으로 뜨므로 URL만 넣으면 됩니다.
- **로컬 GPU** — `HAWKEYE_VLM_BASE_URL=http://127.0.0.1:8000/v1` 형태. API 키 불필요.

## 발표 전 확인할 것

1. **엔드포인트를 미리 켜두세요.** 첫 요청이 느립니다.
2. **백엔드 워밍업** — 재시작 후 첫 요청이 71초 걸립니다(ML 라이브러리 지연 로딩). 발표 직전 아무 요청이나 한 번 보내세요.
3. **유튜브 실시간 송출** — 데모 화면에 환자 영상이 나가면 그대로 공개됩니다. 스켈레톤 오버레이나 합성 클립으로 대체하거나, 주최측에 비공개 요청하세요.
4. **폴백 확인** — 엔드포인트를 꺼놓고 한 번 돌려보세요. 기존 분석이 정상으로 나와야 합니다.

## 영빈쌤이 알려주실 것

모델이 준비되면 **엔드포인트 URL과 모델 이름** 두 개만 주시면 됩니다. 나머지는 설정으로 끝납니다.

출력 형식을 위 JSON에 맞추기 어려우시면 알려주세요 — 서버 쪽 파서를 모델 출력에 맞추는 편이 빠를 수 있습니다.
