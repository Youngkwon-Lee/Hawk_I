"""Research QA for V2 event fusion. Does not alter clinical gait metrics."""
import numpy as np
import pytest
from services.gait_cycle_analyzer_v2 import GaitCycleAnalyzerV2, GaitEvent


def event(time, side="L", kind="HS"):
    return GaitEvent(frame_idx=round(time * 30), timestamp=time, event_type=kind, side=side)


def fuse(methods):
    analyzer = GaitCycleAnalyzerV2(fps=30)
    return analyzer._combine_detection_results(methods, np.arange(90) / 30)[0]


@pytest.mark.xfail(strict=True, reason="Known: interleaved other-side event prevents same-side HS clustering within 100ms")
def test_interleaved_other_event_does_not_split_same_side_hs():
    events = fuse({"zeni": [event(0.20), event(0.26)], "velocity": [event(0.23, "R", "TO")]})
    left_hs = [e for e in events if e.event_type == "HS" and e.side == "L"]
    assert len(left_hs) == 1


def test_same_side_hs_merge_without_interleaving():
    events = fuse({"zeni": [event(0.20)], "velocity": [event(0.26)]})
    assert len([e for e in events if e.event_type == "HS" and e.side == "L"]) == 1


def test_opposite_sides_and_event_types_remain_distinct():
    events = fuse({"zeni": [event(0.20, "L", "HS")], "velocity": [event(0.23, "R", "HS"), event(0.24, "L", "TO")]})
    assert {(e.side, e.event_type) for e in events} == {("L", "HS"), ("R", "HS"), ("L", "TO")}
