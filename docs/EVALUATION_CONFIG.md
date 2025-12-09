# VLM 평가 설정 명세서

**작성일**: 2025-11-24
**데이터셋**: 사용자 라벨링 테스트 셋 (485 samples)

---

## 1. 테스트 셋 정확한 통계 (검증 완료)

### 1.1 전체 통계
- **총 샘플**: 485개
- **총 Subjects**: 18명
- **Score 범위**: 0-2 (Score 3-4 없음)

### 1.2 Task별 분포

| Task | Samples | Subjects | Frame Range | Avg Frames |
|------|---------|----------|-------------|------------|
| Gait | 73 | 5 | 585-1147 | 798.4 |
| Finger tapping | 135 | 5 | 129-438 | 178.3 |
| Hand movement | 137 | 5 | 147-632 | 219.9 |
| Leg agility | 140 | 5 | 142-331 | 211.7 |
| **TOTAL** | **485** | **18** | - | - |

**계산 검증**: 73 + 135 + 137 + 140 = 485 ✓

### 1.3 Subject 목록 (검증 완료)

**18명**: 001, 007, 008, 009, 011, 012, 013, 015, 019, 022, 023, 024, 036, 039, 044, 047, 049, 066

**Task별 Subjects**:
- Gait: 007, 011, 036, 039, 044
- Finger tapping: 008, 009, 015, 019, 022
- Hand movement: 009, 013, 023, 024, 049
- Leg agility: 001, 011, 012, 047, 066

### 1.4 Score 분포 (검증 완료)

| Score | Count | Percentage |
|-------|-------|------------|
| 0 | 170 | 35.1% |
| 1 | 280 | 57.7% |
| 2 | 35 | 7.2% |
| **Total** | **485** | **100.0%** |

**계산 검증**: 170 + 280 + 35 = 485 ✓

---

## 2. Qwen 2.5-VL-7B 설정 (참조 기준)

### 2.1 비디오 처리 설정

```python
video_config = {
    "Finger tapping": {
        "fps": 25.0,
        "max_frames": 150,
        "max_pixels": 640 * 480  # 307,200 pixels
    },
    "Hand movement": {
        "fps": 25.0,
        "max_frames": 150,
        "max_pixels": 640 * 480
    },
    "Leg agility": {
        "fps": 20.0,
        "max_frames": 120,
        "max_pixels": 640 * 480
    },
    "Gait": {
        "fps": 4.0,
        "max_frames": 80,
        "max_pixels": 640 * 480
    }
}
```

### 2.2 프롬프트 (Qwen 기준)

```python
def get_prompt(task_name):
    criteria = {
        "Finger tapping": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the finger tapping.",
        "Hand movement": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the hand opening and closing.",
        "Leg agility": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the leg agility (heel stomping).",
        "Gait": "Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing."
    }

    task_criteria = criteria.get(task_name, "Assess the movement quality and signs of Parkinson's disease.")

    prompt = f"""You are an expert neurologist specializing in Parkinson's Disease.
Analyze the video of a patient performing the '{task_name}' task.
{task_criteria}

Rate the severity of the motor impairment on the MDS-UPDRS scale:
0: Normal (No problems)
1: Slight (Slight slowness/small amplitude, no decrement)
2: Mild (Mild slowness/amplitude, some decrement or hesitations)
3: Moderate (Moderate slowness/amplitude, frequent hesitations/halts)
4: Severe (Severe impairment, barely performs the task)

Output ONLY a JSON object with the following format:
{{
  "score": <int, 0-4>,
  "reasoning": "<string, brief explanation>"
}}
"""
    return prompt
```

---

## 3. GPT-4o 설정

### 3.1 프레임 추출 설정

```python
frame_config = {
    "Gait": {"max_frames": 16},
    "Finger tapping": {"max_frames": 12},
    "Hand movement": {"max_frames": 12},
    "Leg agility": {"max_frames": 12}
}
```

**프레임 해상도**: 512x512 (리사이즈 후)

### 3.2 이유

GPT-4o Vision API는:
- 최대 20개 이미지/요청
- 프레임 추출 방식 사용 (비디오 직접 업로드 불가)
- Gait: 긴 비디오 → 16 프레임
- 다른 태스크: 짧은 비디오 → 12 프레임

### 3.3 API 파라미터

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=300,
    temperature=0.0  # 결정론적
)
```

### 3.4 예상 비용
- 샘플당: ~$0.10
- 전체 (485개): **$48.50**

---

## 4. Gemini 2.0 Flash 설정

### 4.1 비디오 업로드 설정

- **직접 비디오 업로드** (프레임 추출 불필요)
- **자동 처리**: Gemini가 내부적으로 처리
- **원본 해상도 유지**

### 4.2 API 파라미터

```python
response = model.generate_content(
    [video_file, prompt],
    generation_config={
        "temperature": 0.0,
        "max_output_tokens": 300,
    }
)
```

### 4.3 예상 비용
- **무료 tier 사용 가능**
- 전체 (485개): **$0**

### 4.4 Rate Limiting
- 1초 대기 (requests 간)
- 예상 시간: 485초 + 처리시간 = 약 15-20분

---

## 5. 통일된 프롬프트 (3개 모델 공통)

```python
def get_prompt(task_name):
    criteria = {
        "Finger tapping": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the finger tapping.",
        "Hand movement": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the hand opening and closing.",
        "Leg agility": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the leg agility (heel stomping).",
        "Gait": "Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing."
    }

    task_criteria = criteria.get(task_name, "Assess the movement quality and signs of Parkinson's disease.")

    prompt = f"""You are an expert neurologist specializing in Parkinson's Disease.
Analyze the video of a patient performing the '{task_name}' task.
{task_criteria}

Rate the severity of the motor impairment on the MDS-UPDRS scale:
0: Normal (No problems)
1: Slight (Slight slowness/small amplitude, no decrement)
2: Mild (Mild slowness/amplitude, some decrement or hesitations)
3: Moderate (Moderate slowness/amplitude, frequent hesitations/halts)
4: Severe (Severe impairment, barely performs the task)

Output ONLY a JSON object with the following format:
{{
  "score": <int, 0-4>,
  "reasoning": "<string, brief explanation>"
}}
"""
    return prompt
```

**동일 프롬프트 사용 이유**: 공정한 비교를 위해

---

## 6. 설정 비교표

| 항목 | Qwen 2.5-VL-7B | GPT-4o | Gemini 2.0 Flash |
|------|----------------|--------|------------------|
| **입력 방식** | 비디오 직접 | 프레임 추출 | 비디오 직접 |
| **Gait fps** | 4.0 | 프레임 16개 | 자동 처리 |
| **Finger/Hand fps** | 25.0 | 프레임 12개 | 자동 처리 |
| **Leg fps** | 20.0 | 프레임 12개 | 자동 처리 |
| **해상도** | 640x480 | 512x512 | 원본 |
| **Temperature** | 0.0 | 0.0 | 0.0 |
| **비용** | GPU | $48.50 | $0 |

---

## 7. 비디오 경로 구조

```
PD4T/PD4T/PD4T/Videos/
├── Gait/
│   └── {patient_id}/{visit_number}.mp4
├── Finger tapping/
│   └── {patient_id}/{visit_number}_{l|r}.mp4
├── Hand movement/
│   └── {patient_id}/{visit_number}_{l|r}.mp4
└── Leg agility/
    └── {patient_id}/{visit_number}_{l|r}.mp4
```

**예시**:
- Gait: `Videos/Gait/039/15-000939.mp4`
- Finger tapping: `Videos/Finger tapping/009/15-001746_r.mp4`

---

## 8. CSV 파일 경로

```
VLM_commercial/data_mapping/
├── gait_test.csv.txt (73 samples)
├── fingertapping_test.csv.txt (135 samples)
├── handmovement_test.csv.txt (137 samples)
└── legagility_test.csv.txt (140 samples)
```

**CSV 형식** (헤더 없음):
```
video_id,frames,score
15-000939_039,585,1
```

---

## 9. 평가 메트릭

### 9.1 주요 지표

1. **Accuracy**: 정확히 일치
   - Baseline (majority class): 57.7%
   - 목표: >65%

2. **MAE**: 평균 절대 오차
   - 목표: <0.30

3. **Weighted Kappa**: 순서형 일치도
   - 목표: >0.60

### 9.2 혼동 행렬 (Confusion Matrix)

예상 형태:
```
        Pred 0  Pred 1  Pred 2
GT 0      ?       ?       ?
GT 1      ?       ?       ?
GT 2      ?       ?       ?
```

---

## 10. 실행 명령어

### GPT-4o
```bash
python scripts/gpt4o_vlm_evaluation.py \
    --base_dir C:/Users/YK/tulip/PD4T/PD4T/PD4T \
    --test_csv_dir C:/Users/YK/tulip/VLM_commercial/data_mapping \
    --api_key [YOUR_OPENAI_KEY]
```

### Gemini 2.0 Flash
```bash
python scripts/gemini_2_flash_evaluation.py \
    --base_dir C:/Users/YK/tulip/PD4T/PD4T/PD4T \
    --test_csv_dir C:/Users/YK/tulip/VLM_commercial/data_mapping \
    --api_key [YOUR_GOOGLE_KEY]
```

### 결과 비교
```bash
python scripts/compare_results.py \
    --gpt4o_results results/gpt4o_results.csv \
    --gemini_results results/gemini_2_flash_results.csv \
    --output results/comparison_report.md
```

---

## 11. 예상 결과

### 11.1 예상 성능

| 모델 | Accuracy | MAE | Kappa | 비용 | 시간 |
|------|----------|-----|-------|------|------|
| GPT-4o | 65-75% | 0.25-0.35 | 0.55-0.70 | $48.50 | 65분 |
| Gemini Flash | 60-70% | 0.30-0.40 | 0.50-0.65 | $0 | 15-20분 |

### 11.2 예상 실행 시간

**GPT-4o**: 485 × 8초 = 3,880초 = **65분**
**Gemini**: 485 × 2초 = 970초 = **16분**

---

## 12. 검증 완료 항목

- [x] 총 샘플 수: 485개
- [x] Subject 수: 18명
- [x] Score 분포: 170 + 280 + 35 = 485
- [x] Task 샘플 수: 73 + 135 + 137 + 140 = 485
- [x] 프롬프트: 3개 모델 동일
- [x] 비디오 경로 구조: 확인
- [x] CSV 파일 위치: 확인

---

**모든 설정 검증 완료. 평가 실행 준비 완료.**
