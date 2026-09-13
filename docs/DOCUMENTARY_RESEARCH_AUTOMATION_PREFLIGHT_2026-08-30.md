# 다큐 연구 자동화 사전점검 — 2026-08-30

이 문서는 실제 운영 상태를 **비식별 집계만으로** 촬영하는 로컬 리허설 절차다.
환자 영상, 환자·clip ID, 파일 경로, 계정, 토큰, 내부 URL은 화면에 표시하지 않는다.

## 촬영에서 정직하게 보여줄 흐름

`라벨링 → 검증된 handoff → 엔지니어 학습·승인 → 모델 추론`

학습은 비용과 검토가 필요한 단계이므로 자동 실행하지 않는다. 현재 serving 모델이
현재 label export로 학습됐다는 해시 증거가 없으면 화면도 `학습 연결: 확인 필요`로
표시해야 한다.

## 2026-08-30 검증 상태

- label handoff timer: 15분마다 활성화
- export: train 284 · validation 73 · test 69 (총 426)
- handoff manifest: 해시 검증 통과
- serving: `hawkeye-c0b-seed42` · C0B endpoint 응답 확인
- 전체 상태: `connected_unbound`
  - 의미: 라벨 export와 현재 모델 추론은 실제로 연결되어 있으나, 이 export와
    현재 serving 모델의 학습 이력 해시 연결은 선언되지 않았다.

## 촬영 직전 로컬 리허설

### 1. HawkEye 분석 화면

Hawk I `frontend`에서 아래를 실행한다.

```bash
pnpm dev
```

브라우저에서 `http://localhost:3000/filming-demo`를 연다. 이 화면은 공개 Pexels
영상과 같은 프레임에서 만든 관절 추적·각도·속도 데이터만 사용한다.

### 2. 실제 파이프라인 상태 화면

첫 터미널에서 홈 데스크톱 backend로 임시 loopback tunnel을 만든다.

```bash
ssh -N -L 127.0.0.1:5891:127.0.0.1:5891 yk@100.125.26.99
```

두 번째 터미널에서 Kinelo Ops를 실제 aggregate 상태 API에 연결한다.

```bash
cd /Users/youngkwon/projects/kinelo-ops
HAWKEYE_PIPELINE_STATUS_URL=http://127.0.0.1:5891/api/pipeline/status \
  pnpm exec next dev --port 3001
```

브라우저에서 `http://localhost:3001/filming-demo`를 연다.

화면에는 export 건수, handoff 검증, 학습 연결 상태, model endpoint 응답만 표시된다.
`학습 연결`이 `확인 필요`이면 그 상태 그대로 촬영한다. 완료된 학습처럼 보이게
표시하거나 내레이션하지 않는다.

### 3. 촬영 순서

1. HawkEye 원본·관절 추적 비교: 5초
2. Kinelo Ops live pipeline status: 3초
3. Notion의 네 단계 협업 흐름: 3초

## 종료와 재확인

- Kinelo Ops dev server와 SSH tunnel은 촬영 후 종료한다.
- 실제 환자 화면, 라벨링 원본, shell·주소창, 계정 영역은 촬영하지 않는다.
- 공개 Vercel 주소의 `/filming-demo`는 현재 준비된 로컬 화면과 다를 수 있으므로,
  촬영 전에는 반드시 위 로컬 URL을 다시 열어 확인한다.
