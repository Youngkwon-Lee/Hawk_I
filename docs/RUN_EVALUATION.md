# VLM 평가 실행 가이드

**최종 버전**: v2 (data_mapping 사용)
**테스트 셋**: 485 samples, 18 subjects
**작성일**: 2025-11-24

---

## 준비 사항 확인

### 1. 데이터 확인
```bash
# 테스트 CSV 파일 확인 (485개)
wc -l VLM_commercial/data_mapping/*.txt
# 출력: 73 + 135 + 137 + 140 = 485

# 비디오 파일 경로 확인
ls PD4T/PD4T/PD4T/Videos/Gait/039/
ls "PD4T/PD4T/PD4T/Videos/Finger tapping/009/"
```

### 2. API 키 확인
```bash
# API 키 확인
cat VLM_commercial/configs/api_keys.env

# 환경변수 설정 (선택)
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### 3. Python 패키지 확인
```bash
cd VLM_commercial
pip install -r requirements.txt

# 필수 패키지:
# - openai>=1.10.0
# - google-generativeai>=0.3.0
# - opencv-python>=4.8.0
# - pandas>=2.0.0
# - tqdm
```

---

## GPT-4o 평가 실행

### 기본 실행
```bash
cd C:/Users/YK/tulip/VLM_commercial

python scripts/gpt4o_vlm_evaluation_v2.py
```

### 명시적 경로 지정
```bash
python scripts/gpt4o_vlm_evaluation_v2.py \
    --base_dir C:/Users/YK/tulip/PD4T/PD4T/PD4T \
    --test_csv_dir C:/Users/YK/tulip/VLM_commercial/data_mapping \
    --api_key YOUR_OPENAI_API_KEY
```

### 예상 출력
```
OpenAI client initialized
Using test CSV directory: C:/Users/YK/tulip/VLM_commercial/data_mapping
Using video directory: C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos

Processing Gait: 73 samples
Gait: 100%|████████| 73/73 [09:44<00:00, 8.00s/it]

Processing Finger tapping: 135 samples
Finger tapping: 100%|████████| 135/135 [18:00<00:00, 8.00s/it]

...

Results saved to C:/Users/YK/tulip/VLM_commercial/results/gpt4o_results.csv

================================================================================
Evaluation Results
================================================================================
Total samples processed: 485
Valid predictions: 485
Failed predictions: 0

Accuracy: 0.XXX
MAE: 0.XXX

Per-task breakdown:
  Gait: Acc=0.XXX, MAE=0.XXX, N=73
  Finger tapping: Acc=0.XXX, MAE=0.XXX, N=135
  Hand movement: Acc=0.XXX, MAE=0.XXX, N=137
  Leg agility: Acc=0.XXX, MAE=0.XXX, N=140
```

### 예상 실행 시간 및 비용
- **시간**: 485개 × 8초 = 약 65분
- **비용**: 485개 × $0.10 = 약 $48.50

---

## Gemini 2.0 Flash 평가 실행

### 기본 실행
```bash
cd C:/Users/YK/tulip/VLM_commercial

python scripts/gemini_2_flash_evaluation_v2.py
```

### 명시적 경로 지정
```bash
python scripts/gemini_2_flash_evaluation_v2.py \
    --base_dir C:/Users/YK/tulip/PD4T/PD4T/PD4T \
    --test_csv_dir C:/Users/YK/tulip/VLM_commercial/data_mapping \
    --api_key YOUR_GOOGLE_API_KEY
```

### 예상 출력
```
Gemini 2.0 Flash model initialized
Using test CSV directory: C:/Users/YK/tulip/VLM_commercial/data_mapping
Using video directory: C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos

Processing Gait: 73 samples
Gait: 100%|████████| 73/73 [02:26<00:00, 2.00s/it]

Processing Finger tapping: 135 samples
Finger tapping: 100%|████████| 135/135 [04:30<00:00, 2.00s/it]

...

Results saved to C:/Users/YK/tulip/VLM_commercial/results/gemini_2_flash_results.csv

================================================================================
Evaluation Results
================================================================================
Total samples processed: 485
Valid predictions: 485
Failed predictions: 0

Accuracy: 0.XXX
MAE: 0.XXX

Per-task breakdown:
  Gait: Acc=0.XXX, MAE=0.XXX, N=73
  Finger tapping: Acc=0.XXX, MAE=0.XXX, N=135
  Hand movement: Acc=0.XXX, MAE=0.XXX, N=137
  Leg agility: Acc=0.XXX, MAE=0.XXX, N=140
```

### 예상 실행 시간 및 비용
- **시간**: 485개 × 2초 = 약 16분
- **비용**: $0 (무료 tier)

---

## 결과 파일 확인

### 출력 파일 위치
```
VLM_commercial/results/
├── gpt4o_results.csv          # GPT-4o 전체 결과
├── gemini_2_flash_results.csv # Gemini 전체 결과
├── gpt4o_results_temp.csv     # GPT-4o 중간 저장 (10개마다)
└── gemini_results_temp.csv    # Gemini 중간 저장 (10개마다)
```

### CSV 형식
```csv
task,filename,gt_score,pred_score,reason,raw_output
Gait,15-000939_039,1,1,"Slight reduction in stride...","{\"score\": 1, ...}"
```

### 결과 확인
```bash
# 샘플 수 확인
wc -l VLM_commercial/results/gpt4o_results.csv
# 출력: 486 (헤더 1 + 데이터 485)

# 처음 5개 확인
head -6 VLM_commercial/results/gpt4o_results.csv

# Score 분포 확인
python << 'EOF'
import pandas as pd
df = pd.read_csv('VLM_commercial/results/gpt4o_results.csv')
print("GT Score 분포:")
print(df['gt_score'].value_counts().sort_index())
print("\nPred Score 분포:")
print(df['pred_score'].value_counts().sort_index())
print(f"\nAccuracy: {(df['pred_score'] == df['gt_score']).mean():.3f}")
EOF
```

---

## 결과 비교 (GPT-4o vs Gemini)

### 비교 스크립트 실행
```bash
python scripts/compare_results.py \
    --gpt4o_results VLM_commercial/results/gpt4o_results.csv \
    --gemini_results VLM_commercial/results/gemini_2_flash_results.csv \
    --output VLM_commercial/results/comparison_report.md
```

### 출력
```
Loading results...
GPT-4o valid samples: 485
Gemini valid samples: 485

Computing overall metrics...
Computing task-wise metrics...
Generating comparison report...

Report saved to: VLM_commercial/results/comparison_report.md

================================================================================
COMPARISON SUMMARY
================================================================================
GPT-4o Accuracy: 0.XXX | MAE: 0.XXX | Kappa: 0.XXX
Gemini Accuracy: 0.XXX | MAE: 0.XXX | Kappa: 0.XXX
================================================================================
```

---

## 문제 해결

### 1. 비디오 파일을 찾을 수 없음
```
Error: Video not found: PD4T/PD4T/PD4T/Videos/Gait/039/15-000939.mp4
```

**해결**:
- 비디오 파일 경로 확인
- 파일명 확인 (CSV의 video_id와 일치해야 함)

### 2. API 키 오류
```
Error: OpenAI API key not provided
```

**해결**:
```bash
# 환경변수 설정
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."

# 또는 명령어에 직접 지정
python scripts/gpt4o_vlm_evaluation_v2.py --api_key "sk-..."
```

### 3. Rate Limit 초과
```
API Error: Rate limit exceeded
```

**해결**:
- GPT-4o: 스크립트에 자동 재시도 없음 → 잠시 대기 후 재실행
- Gemini: 1초 대기 있음 → 문제 없음

### 4. 메모리 부족
```
Error: Out of memory
```

**해결**:
- 프레임 해상도 축소: `cv2.resize(frame, (256, 256))` (현재 512x512)
- max_frames 줄이기: Gait 16 → 12

---

## 체크리스트

평가 실행 전:
- [ ] 테스트 CSV 파일 존재 확인 (485개)
- [ ] 비디오 파일 경로 확인
- [ ] API 키 설정 확인
- [ ] Python 패키지 설치 확인
- [ ] 충분한 저장 공간 확인 (결과 CSV ~1MB)

평가 실행 중:
- [ ] 진행률 표시 확인 (tqdm)
- [ ] 중간 저장 확인 (10개마다)
- [ ] 에러 메시지 모니터링

평가 완료 후:
- [ ] 결과 CSV 샘플 수 확인 (485개)
- [ ] Accuracy/MAE 합리적인지 확인
- [ ] Failed predictions 개수 확인

---

## 추가 분석

### Confusion Matrix 생성
```python
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

df = pd.read_csv('VLM_commercial/results/gpt4o_results.csv')
df_valid = df[df['pred_score'] >= 0]

cm = confusion_matrix(df_valid['gt_score'], df_valid['pred_score'], labels=[0,1,2])
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(df_valid['gt_score'], df_valid['pred_score'], labels=[0,1,2]))
```

### 오답 케이스 분석
```python
df = pd.read_csv('VLM_commercial/results/gpt4o_results.csv')
errors = df[df['pred_score'] != df['gt_score']]

print(f"Total errors: {len(errors)}")
print("\nError breakdown:")
for task in errors['task'].unique():
    task_errors = errors[errors['task'] == task]
    print(f"  {task}: {len(task_errors)} errors")

# 가장 큰 오차 케이스
df['error'] = (df['pred_score'] - df['gt_score']).abs()
worst_cases = df.nlargest(10, 'error')
print("\nWorst 10 predictions:")
print(worst_cases[['task', 'filename', 'gt_score', 'pred_score', 'reason']])
```

---

**모든 준비 완료. 평가를 시작하세요!**
