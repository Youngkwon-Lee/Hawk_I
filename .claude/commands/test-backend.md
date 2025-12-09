# Backend Test Command

백엔드 테스트 실행 워크플로우

## 실행 옵션

```bash
# 전체 테스트
cd backend && python -m pytest tests/ -v

# 서비스 테스트만
python -m pytest tests/test_services.py -v

# 에이전트 테스트만
python -m pytest tests/test_agents.py -v

# 커버리지 포함
python -m pytest tests/ -v --cov=services --cov=agents --cov-report=term-missing

# 특정 테스트 클래스
python -m pytest tests/test_services.py::TestMLScorer -v

# 특정 테스트 함수
python -m pytest tests/test_services.py::TestMLScorer::test_predict_finger_tapping_rf -v
```

## 테스트 구조

```
backend/tests/
├── conftest.py      # Fixtures (mock data)
├── test_services.py # MetricsCalculator, MLScorer, UPDRSScorer, Visualization
└── test_agents.py   # Clinical, ModelSelector, Validation, Report agents
```

## 주의사항
- mediapipe 필요 테스트는 자동 skip
- 실제 비디오 필요 테스트는 자동 skip
- OpenAI API 없이도 테스트 가능 (fallback 테스트)
