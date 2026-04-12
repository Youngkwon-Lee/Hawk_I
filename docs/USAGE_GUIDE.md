# VLM Commercial Models - Quick Usage Guide

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
# 디렉토리 이동
cd C:/Users/YK/tulip/VLM_commercial

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 확인

API 키는 이미 `configs/api_keys.env`에 설정되어 있습니다:
- OpenAI GPT-4o: ✅ 설정됨
- Google Gemini 2.0 Flash: ✅ 설정됨

### 3. 평가 실행

#### Option A: 모든 태스크 평가

```bash
# GPT-4o 전체 평가
python scripts/gpt4o_vlm_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --api_key $(grep OPENAI_API_KEY configs/api_keys.env | cut -d '=' -f2)

# Gemini 2.0 Flash 전체 평가
python scripts/gemini_2_flash_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --api_key $(grep GOOGLE_API_KEY configs/api_keys.env | cut -d '=' -f2)
```

#### Option B: 특정 태스크만 평가

```bash
# GPT-4o - Gait만 평가
python scripts/gpt4o_vlm_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --task "Gait"

# Gemini - Finger tapping만 평가
python scripts/gemini_2_flash_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --task "Finger tapping"
```

### 4. 결과 비교

```bash
python scripts/compare_results.py \
    --gpt4o_results results/gpt4o_results.csv \
    --gemini_results results/gemini_2_flash_results.csv \
    --output results/comparison_report.md
```

---

## 📊 예상 결과

### 평가 지표

| Metric | Description |
|--------|-------------|
| **Accuracy** | 정확히 일치하는 예측 비율 |
| **MAE** | 평균 절대 오차 (낮을수록 좋음) |
| **Weighted Kappa** | 순서형 데이터 일치도 (높을수록 좋음) |

### 태스크별 샘플 수

| Task | Test Samples |
|------|--------------|
| Gait | 74 |
| Finger tapping | 136 |
| Hand movement | 227 |
| Leg agility | 224 |
| **Total** | **661** |

---

## 💰 비용 추정

### GPT-4o
- **입력**: 이미지당 ~$0.01
- **예상 총 비용**: $50-100 (전체 테스트셋)
- **샘플당**: ~$0.08-0.15

### Gemini 2.0 Flash
- **입력**: 무료 tier 사용 가능
- **예상 총 비용**: $0 (무료 한도 내)
- **제한**: 분당 60 요청

---

## 📁 출력 파일

### 1. 평가 결과 CSV

**위치**: `results/gpt4o_results.csv`, `results/gemini_2_flash_results.csv`

**형식**:
```csv
task,filename,gt_score,pred_score,reason,raw_output
Gait,15-000939_039,1,1,"Patient shows slight slowness...","{\"score\": 1, \"reasoning\": \"...\"}"
```

### 2. 비교 리포트

**위치**: `results/comparison_report.md`

**내용**:
- 전체 성능 비교
- 태스크별 성능
- Confusion Matrix
- Classification Report
- 주요 인사이트

---

## 🔧 문제 해결

### API 키 에러

```bash
# 환경변수로 설정
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# 또는 스크립트 실행 시 직접 지정
python scripts/gpt4o_vlm_evaluation.py --api_key "your-key"
```

### Rate Limit 초과

- GPT-4o: 요청 간 대기 시간 추가 (스크립트에 이미 구현됨)
- Gemini: `time.sleep(1)` 사용 (스크립트에 이미 구현됨)

### 비디오 처리 실패

- `pred_score = -1`인 샘플은 처리 실패
- 로그 파일 확인: `logs/` 디렉토리

---

## 📈 평가 프로세스

```
1. 비디오 로드
   ├─ GPT-4o: 프레임 추출 (최대 16프레임)
   └─ Gemini: 비디오 직접 업로드

2. API 호출
   ├─ GPT-4o: Vision API (이미지 배열)
   └─ Gemini: Multimodal API (비디오 파일)

3. 응답 파싱
   └─ JSON 형식: {"score": 0-4, "reasoning": "..."}

4. 결과 저장
   └─ CSV 파일 + 중간 체크포인트
```

---

## 🎯 다음 단계

### 1. 결과 분석
```bash
# 리포트 확인
cat results/comparison_report.md

# CSV 데이터 분석
python -c "import pandas as pd; df = pd.read_csv('results/gpt4o_results.csv'); print(df.describe())"
```

### 2. 오류 케이스 분석
```python
import pandas as pd

# 오답 케이스 필터링
df = pd.read_csv('results/gpt4o_results.csv')
errors = df[df['pred_score'] != df['gt_score']]
print(errors[['task', 'filename', 'gt_score', 'pred_score', 'reason']])
```

### 3. Prompt 최적화
- `scripts/gpt4o_vlm_evaluation.py`의 `get_prompt()` 함수 수정
- A/B 테스트로 성능 비교

---

## 📞 지원

- **PD4T 데이터셋**: a.dadashzadeh@bristol.ac.uk
- **OpenAI API**: https://platform.openai.com/docs
- **Google Gemini API**: https://ai.google.dev/docs

---

**마지막 업데이트**: 2025-11-24
**버전**: 1.0.0
