# Data Validation Command

PD4T/TULIP 데이터셋 검증 워크플로우

## 실행
```bash
# 전체 검증
python scripts/data_validator.py

# 특정 태스크만
python scripts/data_validator.py --task "Finger Tapping"
```

## 검증 항목
1. **Patient-Level Leakage**: Train/Valid/Test 간 환자 중복 체크
2. **Score Distribution**: 점수 분포 확인
3. **Sample Counts**: 적절한 분할 비율 확인

## 예상 출력
```
============================================================
Validating: Finger Tapping
============================================================

Annotations: ['train', 'test']
  train: 590 videos, 308 patients
  test: 216 videos, 112 patients

Features: ['train', 'valid', 'test']
  train: 568 samples, 293 patients
  valid: 103 samples, 55 patients
  test: 135 samples, 72 patients

============================================================
VALIDATION SUMMARY
============================================================

[ERRORS]
  [CRITICAL] Train-Test overlap: 1 patients

[OK] 에러가 없으면 학습 진행 가능
```

## 주의사항
- 학습 전 항상 이 검증을 먼저 실행할 것
- CRITICAL 에러가 있으면 데이터 재분할 필요
- WARNING은 참고 사항
