# Implementation Plan - Visual Feature Enhancement

# Goal
Implement Trajectory Maps (Temporal Maps) and refine Heatmap generation to provide better visual evidence for the VLM and the Doctor.

## Proposed Changes

### Backend

#### [MODIFY] [vision_agent.py](file:///c:/Users/YK/tulip/Hawkeye/backend/agents/vision_agent.py)
- Import `VisualizationService`.
- Initialize `VisualizationService` in `__init__`.
- In `process()`:
    - After Skeleton Extraction, call `visualization_service.generate_temporal_map`.
    - Pass `ctx.skeleton_data` (need to ensure format matches).
    - Store `trajectory_map_path` in `ctx.vision_meta`.

#### [MODIFY] [visualization.py](file:///c:/Users/YK/tulip/Hawkeye/backend/services/visualization.py)
- Review `generate_temporal_map` to ensure it handles the data format provided by `VisionAgent`.
- Ensure it saves to the correct location.

#### [MODIFY] [context.py](file:///c:/Users/YK/tulip/Hawkeye/backend/domain/context.py)
- (Optional) Verify `vision_meta` dictionary structure is flexible enough (it seems to be a Dict, so likely no change needed, but good to check).

## Verification Plan

### Automated Tests
- Run `app.py` and hit the `/api/analyze` endpoint with a test video.
- Check if `_trajectory.jpg` is generated in the video directory.
- Check if the JSON response includes `trajectory_map_path`.

### Manual Verification
- Inspect the generated images to ensure they look correct (lines connecting joints).
