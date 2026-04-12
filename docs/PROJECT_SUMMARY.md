# VLM Commercial Models Evaluation - Project Summary

## 📋 프로젝트 개요

**목적**: OpenAI GPT-4o와 Google Gemini 2.0 Flash를 사용하여 PD4T 파킨슨병 데이터셋의 운동 품질을 평가하고 성능을 비교

**데이터셋**: PD4T (Parkinson's Disease 4 Tasks)
- 전체 비디오: 2,931개
- 테스트 샘플: 661개
- 태스크: 4개 (Gait, Finger tapping, Hand movement, Leg agility)
- 라벨: MDS-UPDRS 점수 (0-4)

---

## 🗂️ 프로젝트 구조

```
VLM_commercial/
├── README.md                          # 메인 문서
├── USAGE_GUIDE.md                     # 사용 가이드
├── PROJECT_SUMMARY.md                 # 이 파일
├── requirements.txt                   # Python 의존성
├── .gitignore                         # Git 제외 파일
├── run_evaluation.sh                  # 실행 스크립트
│
├── configs/
│   ├── api_keys.env                  # API 키 (실제 키 포함)
│   └── api_keys.env.example          # API 키 템플릿
│
├── scripts/
│   ├── gpt4o_vlm_evaluation.py       # GPT-4o 평가 스크립트
│   ├── gemini_2_flash_evaluation.py  # Gemini 평가 스크립트
│   └── compare_results.py            # 결과 비교 스크립트
│
├── data_mapping/
│   ├── gait_test.csv.txt             # Gait 테스트 매핑 (74 samples)
│   ├── fingertapping_test.csv.txt    # Finger tapping (136 samples)
│   ├── handmovement_test.csv.txt     # Hand movement (227 samples)
│   └── legagility_test.csv.txt       # Leg agility (224 samples)
│
├── models/                            # 모델 설정 (향후 확장)
├── results/                           # 평가 결과 저장
└── logs/                              # 로그 파일
```

---

## 🔑 주요 구성 요소

### 1. GPT-4o 평가 스크립트

**파일**: `scripts/gpt4o_vlm_evaluation.py`

**특징**:
- 비디오에서 프레임 추출 (최대 16프레임)
- OpenAI Vision API 사용
- Base64 인코딩으로 이미지 전송
- JSON 응답 파싱

**설정**:
```python
frame_config = {
    "Gait": {"max_frames": 16},
    "Finger tapping": {"max_frames": 12},
    "Hand movement": {"max_frames": 12},
    "Leg agility": {"max_frames": 12}
}
```

### 2. Gemini 2.0 Flash 평가 스크립트

**파일**: `scripts/gemini_2_flash_evaluation.py`

**특징**:
- 비디오 파일 직접 업로드
- Google Generative AI API 사용
- 자동 파일 삭제 (스토리지 관리)
- Rate limiting (1초 대기)

**장점**:
- 무료 tier 사용 가능
- 비디오 전체 컨텍스트 활용

### 3. 결과 비교 스크립트

**파일**: `scripts/compare_results.py`

**기능**:
- 전체 성능 지표 계산
- 태스크별 성능 분석
- Confusion Matrix 생성
- Markdown 리포트 자동 생성

**지표**:
- Accuracy, MAE, Weighted Kappa
- Precision, Recall, F1-Score
- Classification Report

---

## 🚀 실행 방법

### Method 1: 개별 실행

```bash
# GPT-4o만 실행
python scripts/gpt4o_vlm_evaluation.py --base_dir ../PD4T/PD4T/PD4T

# Gemini만 실행
python scripts/gemini_2_flash_evaluation.py --base_dir ../PD4T/PD4T/PD4T

# 결과 비교
python scripts/compare_results.py \
    --gpt4o_results results/gpt4o_results.csv \
    --gemini_results results/gemini_2_flash_results.csv
```

### Method 2: 배치 스크립트

```bash
# 모든 모델 실행 및 비교
./run_evaluation.sh both

# GPT-4o만 실행
./run_evaluation.sh gpt4o

# Gemini만 실행
./run_evaluation.sh gemini

# 특정 태스크만 평가
./run_evaluation.sh both "Gait"
```

---

## 📊 예상 출력

### 1. CSV 결과 파일

**예시**: `results/gpt4o_results.csv`

```csv
task,filename,gt_score,pred_score,reason,raw_output
Gait,15-000939_039,1,1,"Slight reduction in stride amplitude...","{\"score\": 1, ...}"
Gait,14-005971_039,0,0,"Normal gait pattern observed...","{\"score\": 0, ...}"
```

### 2. 비교 리포트

**예시**: `results/comparison_report.md`

```markdown
# VLM Comparison Report

| Metric | GPT-4o | Gemini | Winner |
|--------|--------|--------|--------|
| Accuracy | 0.75 | 0.72 | GPT-4o |
| MAE | 0.35 | 0.42 | GPT-4o |
| Kappa | 0.68 | 0.65 | GPT-4o |
```

---

## 💡 주요 설계 결정

### 1. 프레임 샘플링 vs 전체 비디오

- **GPT-4o**: 프레임 샘플링 (API 제약)
  - 장점: 비용 효율적, 빠른 처리
  - 단점: 시간적 컨텍스트 손실

- **Gemini**: 전체 비디오 업로드
  - 장점: 완전한 시간적 컨텍스트
  - 단점: 업로드/처리 시간 증가

### 2. Prompt 설계

**공통 구조**:
1. 역할 설정: "You are an expert neurologist..."
2. 태스크별 평가 기준
3. MDS-UPDRS 스케일 설명
4. JSON 출력 형식 강제

**효과**:
- 일관된 응답 형식
- 구조화된 reasoning
- 쉬운 파싱

### 3. 에러 처리

- API 실패 시 `pred_score = -1`
- 중간 저장 (10샘플마다)
- Rate limiting 자동 적용

---

## 📈 성능 최적화

### 1. 토큰 사용 최적화

- 프레임 리사이즈 (512x512)
- 태스크별 프레임 수 조정
- Temperature = 0.0 (결정론적)

### 2. 비용 절감

- Gemini 무료 tier 우선 사용
- 태스크별 선택적 평가
- 중간 결과 재사용 가능

### 3. 속도 개선

- 병렬 처리 가능 (향후 확장)
- 캐싱 전략 (프레임 추출)

---

## 🔮 향후 개선 방향

### 1. 앙상블 방법론
- GPT-4o + Gemini 예측 결합
- Voting 또는 Weighted average

### 2. Prompt Engineering
- Chain-of-Thought prompting
- Few-shot examples 추가
- 도메인 지식 강화

### 3. 평가 지표 확장
- Per-patient consistency
- Temporal coherence analysis
- Reasoning quality assessment

### 4. 인프라 개선
- 배치 처리 최적화
- 분산 실행 지원
- 모니터링 대시보드

---

## 📝 체크리스트

### ✅ 완료된 작업

- [x] PD4T 데이터셋 분석 및 이해
- [x] GPT-4o 평가 스크립트 작성
- [x] Gemini 2.0 Flash 평가 스크립트 작성
- [x] 결과 비교 및 분석 스크립트 작성
- [x] 프로젝트 구조 설계 및 구현
- [x] API 키 설정 및 관리
- [x] 문서화 (README, USAGE_GUIDE, PROJECT_SUMMARY)
- [x] 배치 실행 스크립트 작성
- [x] Git 설정 (.gitignore)

### 🔄 진행 중

- [ ] 실제 평가 실행 및 결과 수집
- [ ] 결과 분석 및 인사이트 도출

### 📋 다음 단계

- [ ] GPT-4o 전체 평가 실행
- [ ] Gemini 전체 평가 실행
- [ ] 결과 비교 및 리포트 생성
- [ ] 오류 케이스 분석
- [ ] Prompt 최적화 실험
- [ ] 논문/보고서 작성

---

## 📞 연락처

- **프로젝트 담당**: YK
- **PD4T 데이터셋**: a.dadashzadeh@bristol.ac.uk
- **OpenAI 지원**: https://platform.openai.com/docs
- **Google Gemini 지원**: https://ai.google.dev/docs

---

**프로젝트 생성일**: 2025-11-24
**마지막 업데이트**: 2025-11-24
**버전**: 1.0.0
**상태**: ✅ 준비 완료 (Ready for Evaluation)
