"""Synthetic-only regression coverage for whole-clip motion ROI sampling."""
import cv2
import numpy as np
import pytest

from services.roi_detector import MovementBasedROI


def _video(tmp_path, frame_count=400, fps=80, moving_start=280):
    path = tmp_path / "synthetic_movement.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64)
    )
    assert writer.isOpened()
    try:
        for idx in range(frame_count):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            if idx >= moving_start:
                left = 10 + ((idx - moving_start) // 2) % 65
                cv2.rectangle(frame, (left, 20), (left + 10, 35),
                              (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


def test_motion_only_near_end_of_5s_80fps_clip_is_seen(tmp_path):
    video = _video(tmp_path)
    movement_map = MovementBasedROI()._calculate_movement_map(
        video, num_frames=30
    )
    assert movement_map.shape == (64, 96)
    assert movement_map.max() == pytest.approx(1.0)
    assert np.count_nonzero(movement_map[18:38, 8:90]) > 0


def test_short_video_is_sampled_without_duplicate_indices(tmp_path):
    video = _video(tmp_path, frame_count=4, fps=80, moving_start=2)
    movement_map = MovementBasedROI()._calculate_movement_map(
        video, num_frames=30
    )
    assert movement_map.shape == (64, 96)
    assert movement_map.max() > 0


def test_static_video_has_no_motion(tmp_path):
    video = _video(tmp_path, frame_count=80, moving_start=1000)
    movement_map = MovementBasedROI()._calculate_movement_map(
        video, num_frames=30
    )
    assert np.all(movement_map == 0)
