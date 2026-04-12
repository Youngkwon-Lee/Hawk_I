# Analyze Video Command

단일 영상 분석 CLI 워크플로우

## 사용법

```bash
cd backend

# 단일 영상 분석
python -c "
import sys
sys.path.insert(0, '.')
from agents.orchestrator import OrchestratorAgent
from domain.context import AnalysisContext

# Video path (change as needed)
video_path = 'path/to/your/video.mp4'

# Create context
ctx = AnalysisContext(video_path=video_path)

# Run analysis
orchestrator = OrchestratorAgent()
result = orchestrator.analyze(ctx)

# Print results
if result.error:
    print(f'Error: {result.error}')
else:
    print(f'Task Type: {result.task_type}')
    print(f'UPDRS Score: {result.clinical_scores}')
    print(f'Metrics: {result.kinematic_metrics}')
"
```

## API로 분석

```bash
# 영상 업로드 및 분석 시작
curl -X POST \
  -F "video=@/path/to/video.mp4" \
  -F "task_type=finger_tapping" \
  http://localhost:5000/api/analysis/start

# 결과 확인 (video_id 사용)
curl http://localhost:5000/api/analysis/result/{video_id}
```

## 지원 태스크
- `finger_tapping`: 손가락 두드리기
- `hand_movement`: 손 움직임
- `gait`: 보행
- `leg_agility`: 다리 민첩성

## 출력 파일
- `{video_id}_heatmap.jpg`: Motion heatmap
- `{video_id}_trajectory.jpg`: Trajectory map
- `{video_id}_skeleton.json`: Landmark data
