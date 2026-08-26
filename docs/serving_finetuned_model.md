# 파인튜닝 모델을 웹사이트에 연결하기

작성: 2026-08-07 · 대상: 모델 학습 담당(영빈), 플랫폼 담당(영권)

## 지금 상태

**웹 쪽 연결은 끝났습니다.** 환경변수 두 개만 설정하면 업로드된 영상이 파인튜닝 모델로 분석되고, 결과가 임상 화면에 그대로 뜹니다.

```bash
HAWKEYE_VLM_BASE_URL=https://<엔드포인트>/v1
HAWKEYE_VLM_MODEL=<모델 이름>
HAWKEYE_VLM_API_KEY=<필요한 경우만>
HAWKEYE_VLM_CONDITION=C0B
# llama.cpp 연결 스모크에서는 8프레임부터 확인합니다.
HAWKEYE_VLM_MAX_FRAMES=8
```

설정이 없으면 **아무 일도 일어나지 않습니다** — 기존 분석이 그대로 동작합니다. 엔드포인트가 죽어 있어도 마찬가지로 기존 결과를 냅니다. 데모 중에 GPU가 꺼져도 화면이 깨지지 않습니다.

## 모델이 지켜야 할 출력 형식

OpenAI 호환 `/chat/completions`로 프레임을 받고, 아래 JSON 또는 학습 프롬프트의
`answer: <0-4>` 한 줄 형식으로 반환하면 됩니다.

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

학습 프롬프트와 같이 점수만 반환하는 경우도 허용합니다.

```text
answer: 2
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

## 조건 선택

- `C0B`: 영상만 사용하는 자율 추론 조건입니다. ParkiCheck 자가검사에서 자동 분석할 때
  사용하는 기본 조건입니다.
- `C2B`: 영상과 pose kinematics(K)를 사용합니다. K 블록 생성이 구현된 뒤 활성화합니다.
- `C1BE` / `C3BE`: 임상의가 기록한 구조화 관찰(R)이 입력에 필요합니다. 환자 자가검사에
  자동으로 적용하면 안 되며, 의료진 검토 모드에서만 사용합니다.

백엔드는 현재 `C0B` 학습 계약의 `SYSTEM → ANCHOR → GLOSSARY → QUESTION → VIDEO`
순서를 그대로 전송합니다. 다른 조건을 설정하면 조용히 잘못된 결과를 내지 않고 기존
분석으로 폴백합니다.

## GPU는 어디에 띄우나

**집 데스크톱에는 AMD Radeon RX 6700 XT 12GB GPU가 있습니다.** WSL ROCm 대신 Windows
Vulkan `llama.cpp`를 사용합니다. 로컬 OpenAI 호환 서버는 아래 스크립트로 시작합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_gpu\start_hawkeye_c0b_windows.ps1
```

집 데스크톱 WSL의 Hawk I 백엔드에서는 아래 주소를 사용합니다.

```bash
HAWKEYE_VLM_BASE_URL=http://100.83.147.56:8000/v1
HAWKEYE_VLM_MODEL=hawkeye-c0b-seed42
HAWKEYE_VLM_CONDITION=C0B
```

이 경로는 공식 Qwen Q4 GGUF와 변환한 C0B LoRA를 사용한 웹 연결 스모크입니다. 다중
이미지 입력은 학습 시 Transformers 비디오 프로세서와 동일하지 않으므로 이 결과를
논문 성능 재현값으로 사용하지 않습니다. 성능 재현은 원본 Transformers 환경에서
`5 fps`, `512 px`, `do_sample_frames=False` 계약으로 별도 검증해야 합니다.

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
