import math

from services.gait_quality import periodicity_cadence_reference


def _frames(fps=30, seconds=12, cycle_seconds=1.07, visibility=1.0):
    result = []
    for n in range(int(fps * seconds)):
        x = 0.12 * math.sin(2 * math.pi * n / fps / cycle_seconds)
        result.append({
            "frame_number": n,
            "landmarks": [
                {"id": 27, "x": 0.5 + x / 2, "visibility": visibility},
                {"id": 28, "x": 0.5 - x / 2, "visibility": visibility},
            ],
        })
    return result


def test_periodicity_reference_recovers_stable_approximate_cadence():
    ref = periodicity_cadence_reference(_frames(), 30)
    assert ref is not None
    assert 108 <= ref <= 118


def test_periodicity_reference_rejects_low_visibility():
    assert periodicity_cadence_reference(_frames(visibility=0.1), 30) is None


def test_periodicity_reference_rejects_short_video():
    assert periodicity_cadence_reference(_frames(seconds=2), 30) is None


def test_periodicity_reference_does_not_claim_stationary_signal():
    frames = _frames()
    for frame in frames:
        for point in frame["landmarks"]:
            point["x"] = 0.5
    assert periodicity_cadence_reference(frames, 30) is None


def test_validation_flags_cadence_disagreement_without_rewriting_measurement():
    from agents.validation_agent import ValidationAgent
    from domain.context import AnalysisContext

    ctx = AnalysisContext(video_path="synthetic.mp4")
    ctx.task_type = "gait"
    ctx.vision_meta = {"fps": 30, "frame_count": 360}
    ctx.skeleton_data = {"landmarks": _frames()}
    ctx.kinematic_metrics = {"cadence": 182.0, "stride_length": 0.25}
    result = ValidationAgent()._validate_metrics(ctx)
    assert result["details"]["cadence_disagreement"] is True
    assert 108 <= result["details"]["cadence_periodicity_reference_spm"] <= 118
    assert any("manual review required" in warning for warning in result["warnings"])
    assert ctx.kinematic_metrics["cadence"] == 182.0
