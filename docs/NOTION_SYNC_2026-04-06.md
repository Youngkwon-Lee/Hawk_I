# Hawkeye ↔ Notion 정합 요약 (2026-04-06)

Notion 페이지: `Hawkeye VLM Project 2026`
- URL: https://www.notion.so/Hawkeye-VLM-Project-2026-302336b49bcf81798b0dcecbf5595b65
- 접근: public(anyone) 확인됨

## 1) Notion에서 확인된 상위 구조

- 프로젝트 개요
- 마일스톤
- 회의록
- 팀 구성
- 바로가기

또한 DB(컬렉션) 5개가 연결되어 있음:
- Project Overview
- Milestones Tracker
- Meeting Notes Log
- Rationale Review
- Hawkeye Literature

## 2) 로컬 레포(Hawkeye)와의 매핑

### A. Project Overview ↔ `Hawkeye/docs`, `README.md`
- Notion 개요(연구 목적/상반기·하반기 목표)는 로컬의 아래 문서들과 정렬 필요
  - `README.md`
  - `docs/VLM_TEACHER_API_SELECTION_2026.md`
  - `docs/GEMINI_RATIONALE_PROMPT_GUIDE.md`
  - `docs/VLM_COMMERCIAL_ECOSYSTEM_COMPARISON_2026.md`

### B. Milestones Tracker ↔ 실행 로그/결과물
- 코드 실행 근거: `scripts/`
- 결과물/로그: `results/`, `rationale_results/`
- 권장: 마일스톤 항목마다 아래 3개 필드 고정
  - owner
  - evidence path (repo 상대경로)
  - status date

### C. Meeting Notes Log ↔ `scripts/meetings/`
- 회의 음성/전사 산출물 위치와 연결
- 권장: 회의록 DB에 파일 경로(또는 요약 링크) 필수화

### D. Rationale Review ↔ `backend/rationale_review_app.py`, `rationale_results/`
- 리뷰 앱과 결과 json/csv를 기준으로 검수 루프 추적
- 권장: sample_id/판정결과/재생성여부 컬럼 통일

### E. Hawkeye Literature ↔ `literature_monitor/`, `docs/literature_review.html`
- 문헌 모니터링 산출물과 DB를 주기적으로 동기화

## 3) 지금 바로 실행 가능한 운영 규칙 (추천)

1. Notion의 각 레코드에 "Evidence Path" 필드 추가
   - 예: `rationale_results/v5_rubric/...summary.json`

2. 마일스톤 상태 전환 기준 고정
   - Planned → Running → Review → Done

3. 주간 회고 템플릿 고정
   - 이번주 완료
   - 실패/막힘
   - 다음주 실험 3개

## 4) 빠른 체크리스트

- [ ] Notion Milestones의 각 항목에 repo 경로 연결
- [ ] Rationale Review DB 컬럼과 실제 결과 파일 키 이름 일치화
- [ ] Meeting Notes에 실험 결정사항(파라미터/모델 버전) 누락 방지
- [ ] Literature DB에서 "실험 반영 여부" 컬럼 추가

---
이 문서는 "Notion 운영 구조"와 "현재 레포 구조"의 맞물림을 빠르게 맞추기 위한 1차 동기화 노트입니다.
