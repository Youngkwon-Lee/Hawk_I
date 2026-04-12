# Claude Code Best Practices for ML/DL Research

2025년 12월 기준 커뮤니티 및 공식 가이드 조사 결과

## 핵심 Best Practices

### 1. CLAUDE.md 최적화

**원칙**: Lean Documentation - 간결하고 핵심적인 정보만

```markdown
# 권장 구조
1. 프로젝트 개요 (2-3줄)
2. 폴더 구조 (트리 형식)
3. 주요 명령어 (표 형식)
4. 환경 설정 방법
5. 데이터 경로
```

**피해야 할 것**:
- 너무 긴 설명 (Claude가 무시할 수 있음)
- 중복된 정보
- 자주 변경되는 세부사항

---

### 2. TDD 워크플로우

> "Robots LOVE TDD" - Reddit r/ClaudeAI

**ML/DL에 TDD 적용**:

```python
# 1단계: 테스트 먼저 작성
def test_model_forward():
    model = MyModel(input_dim=34, hidden_dim=128)
    x = torch.randn(16, 100, 34)
    output = model(x)
    assert output.shape == (16, 5)

# 2단계: Claude에게 구현 요청
# "test_model_forward 테스트를 통과하는 MyModel 클래스 구현해줘"
```

**장점**:
- 명확한 요구사항 전달
- 자동 검증 가능
- 디버깅 시간 단축

---

### 3. Custom Commands 활용

`.claude/commands/` 폴더에 워크플로우 저장:

```
.claude/
└── commands/
    ├── train-ml.md      # ML 학습 워크플로우
    ├── train-dl.md      # DL 학습 워크플로우
    ├── eval-vlm.md      # VLM 평가 워크플로우
    └── debug.md         # 디버깅 체크리스트
```

**사용법**: 채팅에서 `/train-ml` 입력하면 해당 워크플로우 실행

---

### 4. MCP 서버 활용

ML/DL 연구에 유용한 MCP 서버:

| 서버 | 용도 | 활용 예시 |
|------|------|----------|
| context7 | 문서 조회 | PyTorch API, TensorFlow 문서 |
| sequential | 복잡한 분석 | 아키텍처 설계, 버그 추적 |
| playwright | 테스트/시각화 | 결과 리포트, 웹 대시보드 |

**활성화 방법**:
```bash
claude mcp add context7
```

---

### 5. 환경별 워크플로우 분리

```
[로컬 환경]              [HPC 환경]
    │                        │
    ├─ 코드 작성              ├─ GPU 학습
    ├─ 소규모 테스트          ├─ 대규모 실험
    ├─ API 기반 VLM          ├─ 로컬 VLM
    └─ 결과 분석              └─ 체크포인트 저장
```

**Claude Code 역할**:
- 로컬: 코드 작성, 디버깅, 분석
- HPC: Job script 생성, 결과 파싱

---

### 6. Hooks 자동화 (고급)

`.claude/hooks/` 로 반복 작업 자동화:

```json
{
  "on_file_save": {
    "pattern": "*.py",
    "command": "python -m pylint {file}"
  },
  "on_train_complete": {
    "command": "python scripts/notify.py"
  }
}
```

---

## 생산성 향상 수치

| 방법 | 예상 향상 | 난이도 |
|------|----------|-------|
| CLAUDE.md 최적화 | 10-15% | 쉬움 |
| TDD 워크플로우 | 20-30% | 중간 |
| Custom Commands | 15-20% | 쉬움 |
| MCP 서버 | 10-20% | 중간 |
| Hooks 자동화 | 10-15% | 어려움 |

**총 예상 향상**: 30-50% (모든 방법 적용 시)

---

## Hawkeye 프로젝트 적용 현황

### 이미 적용됨 ✅
- [x] CLAUDE.md 구조화
- [x] 환경별 설정 (env_config.py)
- [x] Custom Commands 생성
- [x] 폴더 구조 최적화

### 추가 권장사항 💡
- [ ] MCP context7 연동 (PyTorch 문서)
- [ ] 학습 결과 자동 리포트 생성
- [ ] HPC job script 템플릿 확장

---

## 참고 자료

### 공식 문서
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [MCP Server Guide](https://modelcontextprotocol.io)

### 커뮤니티 (2025년 12월 기준)
- Reddit r/ClaudeAI - TDD 워크플로우 토론
- GitHub Discussions - Best practices 공유
- DeepLearning.AI - Claude Code + ML 코스
- DataCamp - AI Coding Assistants 가이드

### 추천 코스
- DeepLearning.AI: "AI Agentic Design Patterns with AutoGen"
- Coursera: "Generative AI for Software Development"
- DataCamp: "Developing AI Systems with the Anthropic SDK"
