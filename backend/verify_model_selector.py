import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.clinical_agent import ClinicalAgent
from domain.context import AnalysisContext
from dataclasses import dataclass

def test_clinical_agent_integration():
    print("Testing Clinical Agent Integration with Model Selector...")
    
    agent = ClinicalAgent()
    
    # Create a mock context
    ctx = AnalysisContext(video_path="test.mp4")
    ctx.task_type = "finger_tapping"
    ctx.status = "vision_done"
    ctx.vision_meta = {"fps": 30.0}
    
    # Mock skeleton data (needs to look real enough to pass verify)
    # Finger tapping needs landmarks 0, 4, 5, 6, 7, 8
    mock_frame = {
        "frame_number": 0,
        "timestamp": 0.0,
        "keypoints": [
            {"id": 0, "x": 0, "y": 0, "z": 0},
            {"id": 4, "x": 0, "y": 0, "z": 0},
            {"id": 5, "x": 0, "y": 0, "z": 0},
            {"id": 6, "x": 0, "y": 0, "z": 0},
            {"id": 7, "x": 0, "y": 0, "z": 0},
            {"id": 8, "x": 0, "y": 0, "z": 0},
        ]
    }
    # Need at least 10 frames for metrics calculator
    # And we need some movement to get peaks
    frames = []
    for i in range(30):
        f = mock_frame.copy()
        f["frame_number"] = i
        f["timestamp"] = i / 30.0
        # Modulate distance between thumb (4) and index (8)
        import math
        val = 0.1 + 0.05 * math.sin(i * 0.5)
        f["keypoints"] = [
            {"id": 0, "x": 0, "y": 0, "z": 0},
            {"id": 4, "x": 0, "y": 0, "z": 0},
            {"id": 5, "x": 0, "y": 0, "z": 0},
            {"id": 6, "x": 0, "y": 0, "z": 0},
            {"id": 7, "x": 0, "y": 0, "z": 0},
            {"id": 8, "x": val, "y": val, "z": val}, # Move index tip
        ]
        frames.append(f)

    ctx.skeleton_data = {"landmarks": frames}
    
    # Run process
    ctx = agent.process(ctx)
    
    if ctx.error:
        print(f"❌ Test Failed: {ctx.error}")
        with open("my_error.txt", "w") as f:
            f.write(ctx.error)
        return

    print("✅ Clinical Agent processed successfully.")
    print(f"Metrics: {ctx.kinematic_metrics}")
    
    if "speed" in ctx.kinematic_metrics or "tapping_speed" in ctx.kinematic_metrics:
        print("✅ Metrics calculated correctly.")
    else:
        print(f"❌ Metrics missing or keys changed. Keys: {ctx.kinematic_metrics.keys()}")

    if ctx.clinical_scores and "total_score" in ctx.clinical_scores:
        print("✅ UPDRS Scores calculated correctly.")
    else:
        print(f"❌ UPDRS Scores missing. Scores: {ctx.clinical_scores}")

if __name__ == "__main__":
    test_clinical_agent_integration()
