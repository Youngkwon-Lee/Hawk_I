import cv2
import numpy as np

from agents.vision_agent import VisionAgent
from domain.context import AnalysisContext


class _EmptyPose:
    def process_video(self, *args, **kwargs):
        return []


class _NoOpViz:
    def generate_temporal_map(self, **kwargs):
        return None


def test_vision_agent_preserves_real_video_dimensions_and_duration(tmp_path):
    video = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    agent = object.__new__(VisionAgent)
    agent.mp_processor_pose = _EmptyPose()
    agent.visualization_service = _NoOpViz()

    ctx = AnalysisContext(video_path=str(video), manual_test_type="gait")
    result = agent.process(ctx)
    assert result.error is None
    assert result.vision_meta["frame_width"] == 160
    assert result.vision_meta["frame_height"] == 120
    assert result.vision_meta["frame_count"] == 20
    assert abs(result.vision_meta["duration"] - 2.0) < 0.05
