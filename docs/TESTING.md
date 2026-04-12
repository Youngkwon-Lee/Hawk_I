# Hawkeye PD Gait Analysis - Testing Guide

## PD4T 데이터셋 테스트

### 데이터셋 위치
```
C:/Users/YK/tulip/PD4T/PD4T/PD4T/
├── Videos/
│   ├── Gait/          # 426 videos
│   ├── Finger tapping/ # 806 videos
│   ├── Hand movement/  # 848 videos
│   └── Leg agility/    # 851 videos
└── Annotations/
    └── Gait/
        ├── train.csv   # 22 subjects
        └── test.csv    # 8 subjects
```

### Annotation 형식
```
# Gait: (visit_number)_(patient_id), frames, score
14-005197_022,325,0   # Score 0 (Normal)
15-002976_019,664,3   # Score 3 (Severe PD)
```

### 비디오 경로 규칙
```
# Gait: Videos/Gait/{patient_id}/{visit_number}.mp4
C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait/022/14-005197.mp4  # Score 0
C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait/019/15-002976.mp4  # Score 3
```

## 테스트용 비디오 예시

### Score 0 (정상) - Test Set
| Annotation | Patient | Video Path | Frames |
|------------|---------|------------|--------|
| 14-005197_022 | 022 | `Videos/Gait/022/14-005197.mp4` | 325 |
| 15-000258_037 | 037 | `Videos/Gait/037/15-000258.mp4` | 541 |
| 15-009973_037 | 037 | `Videos/Gait/037/15-009973.mp4` | 547 |
| 15-006800_037 | 037 | `Videos/Gait/037/15-006800.mp4` | 558 |

### Score 3 (심각한 PD) - Test Set
| Annotation | Patient | Video Path | Frames |
|------------|---------|------------|--------|
| 15-002976_019 | 019 | `Videos/Gait/019/15-002976.mp4` | 664 |
| 14-005778_019 | 019 | `Videos/Gait/019/14-005778.mp4` | 1017 |
| 15-000690_038 | 038 | `Videos/Gait/038/15-000690.mp4` | 1367 |
| 14-005795_019 | 019 | `Videos/Gait/019/14-005795.mp4` | 2240 |

## 테스트 실행 방법

### Python 환경
```bash
# Python 3.10 사용 (MediaPipe 설치됨)
C:/Users/YK/AppData/Local/Programs/Python/Python310/python.exe
```

### 단일 비디오 테스트
```python
import sys
sys.path.insert(0, 'C:/Users/YK/tulip/Hawkeye/backend')

from dataclasses import asdict
from services.mediapipe_processor import MediaPipeProcessor
from services.metrics_calculator import MetricsCalculator
from services.updrs_scorer import UPDRSScorer

# Score 0 테스트
video_path = 'C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait/022/14-005197.mp4'
expected_score = 0

processor = MediaPipeProcessor(mode='pose')
landmark_frames = processor.process_video(video_path)
frames_dict = [asdict(f) for f in landmark_frames]

calculator = MetricsCalculator(fps=30.0)
metrics = calculator.calculate_gait_metrics(frames_dict)

print(f'Walking Speed: {metrics.walking_speed:.2f} m/s')
print(f'Stride Length: {metrics.stride_length:.2f} m')
print(f'Arm Swing (L): {metrics.arm_swing_amplitude_left*100:.1f} cm')
print(f'Arm Swing (R): {metrics.arm_swing_amplitude_right*100:.1f} cm')
print(f'Arm Asymmetry: {metrics.arm_swing_asymmetry:.1f}%')

scorer = UPDRSScorer()
result = scorer.score_gait(metrics)
print(f'Predicted: {result.total_score}, Expected: {expected_score}')
```

## Feature 기대값 참조

### UPDRS Score 기준 (MDS-UPDRS Part III)
| Score | Walking Speed | Stride Length | Arm Swing | Step Height |
|-------|--------------|---------------|-----------|-------------|
| 0 | >1.2 m/s | >0.5 m | >10 cm | >5 cm |
| 1 | 1.0-1.2 m/s | 0.4-0.5 m | 7-10 cm | 4-5 cm |
| 2 | 0.8-1.0 m/s | 0.3-0.4 m | 5-7 cm | 3-4 cm |
| 3 | 0.5-0.8 m/s | 0.15-0.3 m | 3-5 cm | 2-3 cm |
| 4 | <0.5 m/s | <0.15 m | <3 cm | <2 cm |

## 테스트 결과 (2025-11-28)

### PD4T Score 0 (정상) - 14-005197_022
| Feature | 측정값 | UPDRS Score |
|---------|--------|-------------|
| Walking Speed | **1.71 m/s** | 0 |
| Stride Length | **0.46 m** | 0 |
| Arm Swing (L/R) | **6.9 / 5.9 cm** | 1-2 |
| Step Height | **4.8 cm** | 1 |
| Arm Asymmetry | **13.9%** | 0 |

- **예측: 2.3 (Mild) | 기대: 0**
- Base Score: 1, Penalties: 1.3
- 비고: Arm Swing 측정값이 PD4T 환경에서 낮게 나옴 (카메라 거리/각도 영향)

### PD4T Score 3 (심각한 PD) - 15-002976_019
| Feature | 측정값 | UPDRS Score |
|---------|--------|-------------|
| Walking Speed | **0.43 m/s** | 3 |
| Stride Length | **0.16 m** | 3 |
| Arm Swing (L/R) | **4.6 / 6.1 cm** | 3 |
| Step Height | **1.3 cm** | 4 |
| Arm Asymmetry | **23.7%** | 3 |
| Cadence | **162.9 steps/min** | Penalty |

- **예측: 4.0 (Severe) | 기대: 3**
- Base Score: 3, Penalties: 1.0
- 비고: 높은 Cadence는 PD 보상 전략 (짧은 보폭 보상)

### 분석
- **Score 0 vs Score 3 구별**: ✅ 성공
  - Walking Speed: 1.71 → 0.43 m/s (75% 감소)
  - Step Height: 4.8 → 1.3 cm (73% 감소)
- **절대 점수 정확도**: ⚠️ 조정 필요
  - Arm Swing 임계값 하향 조정 필요 (PD4T 환경 특성)
  - Penalty cap 적용 고려

### demo_videos 테스트 결과 (참고)
#### Score 1 영상 (demo_videos/score_1_14-005690_demo.mp4)
- Walking Speed: **1.85 m/s** (정상)
- Stride Length: **0.49 m** (정상)
- Arm Swing: L 10.0 / R 10.8 cm (정상)
- Arm Asymmetry: **7.3%** (정상)

#### Score 3 영상 (demo_videos/score_3_13-007586_demo.mp4)
- Walking Speed: **0.62 m/s** (감소)
- Stride Length: **0.15 m** (매우 짧음)
- Arm Swing: L 4.9 / R 4.8 cm (감소)
- Step Height: **2.5 cm** (낮음)
- Arm Asymmetry: 3.2%

→ Features가 Score 1과 Score 3 사이에 명확한 차이를 보임

## Interpolation 기능

`metrics_calculator.py`에 구현됨:
- `VISIBILITY_THRESHOLD = 0.5` - 유효 랜드마크 기준
- `MAX_INTERPOLATION_GAP = 10` - 최대 10프레임 보간
- 필수 랜드마크: 23, 24 (엉덩이), 27, 28 (발목), 15, 16 (손목)

## 주의사항

- PD4T 데이터는 학술 목적으로만 사용 가능
- 재배포 및 수정 금지
- Bristol 대학교 데이터셋
