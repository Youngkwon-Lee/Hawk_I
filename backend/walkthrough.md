# Walkthrough - Visual Feature Enhancement

I have implemented two key visual features to support the Multi-Agent architecture and provide better explainability for the VLM/Doctor.

## 1. Trajectory Maps (Temporal Maps)
**Goal**: Visualize the smoothness and path of movement (e.g., for Finger Tapping or Hand Movement).
**Implementation**:
- Updated `VisionAgent` to extract skeleton data and pass it to `VisualizationService`.
- `VisualizationService` draws lines connecting key joints (Index Finger for hands, Ankles for gait) across frames.
- **Output**: `_trajectory.jpg` in the video folder.

## 2. Motion Heatmaps (Refined)
**Goal**: Visualize the intensity and location of movement.
**Implementation**:
- Refactored `VisionAgent` to use `VisualizationService` for heatmap generation.
- Moved the pixel-based heatmap logic to `VisualizationService.generate_motion_heatmap`.
- Added support for overlaying the heatmap on the original video frame.
- **Output**: `_heatmap.jpg` in the video folder.

## Verification
I verified the logic using mock data tests (`check_viz.py`, `check_heatmap.py`) to ensure the image generation works correctly with the expected data structures.

### Next Steps
- Expose these file paths in the API response (currently stored in `AnalysisContext`).
- Integrate with the Frontend to display these images.
- Proceed to Phase 2: Collaborative Agent Structure.

## Phase 2: Collaborative Agent Structure

### 1. Orchestrator Refactoring
- **Dynamic Routing**: Implemented logic to check vision confidence. If confidence is low (< 0.4), a warning is logged.
- **Partial Failure Handling**: If Vision fails to extract a skeleton, the Clinical Agent is skipped, but the Report Agent is still called to generate a failure report.
- **File**: `backend/agents/orchestrator.py`

### 2. Clinical Agent Enhancement
- **Chart Generation**: Added `generate_charts` method to format kinematic metrics and UPDRS scores into a Markdown table.
- **Context Update**: Added `clinical_charts` field to `AnalysisContext` to store this structured data for the VLM.
- **File**: `backend/agents/clinical_agent.py`, `backend/domain/context.py`
# Walkthrough - Visual Feature Enhancement

I have implemented two key visual features to support the Multi-Agent architecture and provide better explainability for the VLM/Doctor.

## 1. Trajectory Maps (Temporal Maps)
**Goal**: Visualize the smoothness and path of movement (e.g., for Finger Tapping or Hand Movement).
**Implementation**:
- Updated `VisionAgent` to extract skeleton data and pass it to `VisualizationService`.
- `VisualizationService` draws lines connecting key joints (Index Finger for hands, Ankles for gait) across frames.
- **Output**: `_trajectory.jpg` in the video folder.

## 2. Motion Heatmaps (Refined)
**Goal**: Visualize the intensity and location of movement.
**Implementation**:
- Refactored `VisionAgent` to use `VisualizationService` for heatmap generation.
- Moved the pixel-based heatmap logic to `VisualizationService.generate_motion_heatmap`.
- Added support for overlaying the heatmap on the original video frame.
- **Output**: `_heatmap.jpg` in the video folder.

## Verification
I verified the logic using mock data tests (`check_viz.py`, `check_heatmap.py`) to ensure the image generation works correctly with the expected data structures.

### Next Steps
- Expose these file paths in the API response (currently stored in `AnalysisContext`).
- Integrate with the Frontend to display these images.
- Proceed to Phase 2: Collaborative Agent Structure.

## Phase 2: Collaborative Agent Structure

### 1. Orchestrator Refactoring
- **Dynamic Routing**: Implemented logic to check vision confidence. If confidence is low (< 0.4), a warning is logged.
- **Partial Failure Handling**: If Vision fails to extract a skeleton, the Clinical Agent is skipped, but the Report Agent is still called to generate a failure report.
- **File**: `backend/agents/orchestrator.py`

### 2. Clinical Agent Enhancement
- **Chart Generation**: Added `generate_charts` method to format kinematic metrics and UPDRS scores into a Markdown table.
- **Context Update**: Added `clinical_charts` field to `AnalysisContext` to store this structured data for the VLM.
- **File**: `backend/agents/clinical_agent.py`, `backend/domain/context.py`

### 3. Verification
- **Test Script**: Created `check_orchestrator.py` (temporary) to verify:
    - Happy path (high confidence).
    - Low confidence warning.
    - Partial failure (no skeleton -> skip clinical).
- Result: All tests passed.

## Phase 3: VLM Integration

### 1. Interpretation Agent Update
- **Clinical Charts**: Updated `InterpretationAgent` to accept `clinical_charts` (Markdown tables).
- **Prompt Engineering**: Modified the LLM prompt to include these charts under a "## 상세 차트 데이터" section, giving the VLM direct access to quantitative data.
- **File**: `backend/services/interpretation_agent.py`

### 2. Report Agent Update
# Walkthrough - Visual Feature Enhancement

I have implemented two key visual features to support the Multi-Agent architecture and provide better explainability for the VLM/Doctor.

## 1. Trajectory Maps (Temporal Maps)
**Goal**: Visualize the smoothness and path of movement (e.g., for Finger Tapping or Hand Movement).
**Implementation**:
- Updated `VisionAgent` to extract skeleton data and pass it to `VisualizationService`.
- `VisualizationService` draws lines connecting key joints (Index Finger for hands, Ankles for gait) across frames.
- **Output**: `_trajectory.jpg` in the video folder.

## 2. Motion Heatmaps (Refined)
**Goal**: Visualize the intensity and location of movement.
**Implementation**:
- Refactored `VisionAgent` to use `VisualizationService` for heatmap generation.
- Moved the pixel-based heatmap logic to `VisualizationService.generate_motion_heatmap`.
- Added support for overlaying the heatmap on the original video frame.
- **Output**: `_heatmap.jpg` in the video folder.

## Verification
I verified the logic using mock data tests (`check_viz.py`, `check_heatmap.py`) to ensure the image generation works correctly with the expected data structures.

### Next Steps
- Expose these file paths in the API response (currently stored in `AnalysisContext`).
- Integrate with the Frontend to display these images.
- Proceed to Phase 2: Collaborative Agent Structure.

## Phase 2: Collaborative Agent Structure

### 1. Orchestrator Refactoring
- **Dynamic Routing**: Implemented logic to check vision confidence. If confidence is low (< 0.4), a warning is logged.
- **Partial Failure Handling**: If Vision fails to extract a skeleton, the Clinical Agent is skipped, but the Report Agent is still called to generate a failure report.
- **File**: `backend/agents/orchestrator.py`

### 2. Clinical Agent Enhancement
- **Chart Generation**: Added `generate_charts` method to format kinematic metrics and UPDRS scores into a Markdown table.
- **Context Update**: Added `clinical_charts` field to `AnalysisContext` to store this structured data for the VLM.
- **File**: `backend/agents/clinical_agent.py`, `backend/domain/context.py`
# Walkthrough - Visual Feature Enhancement

I have implemented two key visual features to support the Multi-Agent architecture and provide better explainability for the VLM/Doctor.

## 1. Trajectory Maps (Temporal Maps)
**Goal**: Visualize the smoothness and path of movement (e.g., for Finger Tapping or Hand Movement).
**Implementation**:
- Updated `VisionAgent` to extract skeleton data and pass it to `VisualizationService`.
- `VisualizationService` draws lines connecting key joints (Index Finger for hands, Ankles for gait) across frames.
- **Output**: `_trajectory.jpg` in the video folder.

## 2. Motion Heatmaps (Refined)
**Goal**: Visualize the intensity and location of movement.
**Implementation**:
- Refactored `VisionAgent` to use `VisualizationService` for heatmap generation.
- Moved the pixel-based heatmap logic to `VisualizationService.generate_motion_heatmap`.
- Added support for overlaying the heatmap on the original video frame.
- **Output**: `_heatmap.jpg` in the video folder.

## Verification
I verified the logic using mock data tests (`check_viz.py`, `check_heatmap.py`) to ensure the image generation works correctly with the expected data structures.

### Next Steps
- Expose these file paths in the API response (currently stored in `AnalysisContext`).
- Integrate with the Frontend to display these images.
- Proceed to Phase 2: Collaborative Agent Structure.

## Phase 2: Collaborative Agent Structure

### 1. Orchestrator Refactoring
- **Dynamic Routing**: Implemented logic to check vision confidence. If confidence is low (< 0.4), a warning is logged.
- **Partial Failure Handling**: If Vision fails to extract a skeleton, the Clinical Agent is skipped, but the Report Agent is still called to generate a failure report.
- **File**: `backend/agents/orchestrator.py`

### 2. Clinical Agent Enhancement
- **Chart Generation**: Added `generate_charts` method to format kinematic metrics and UPDRS scores into a Markdown table.
- **Context Update**: Added `clinical_charts` field to `AnalysisContext` to store this structured data for the VLM.
- **File**: `backend/agents/clinical_agent.py`, `backend/domain/context.py`

### 3. Verification
- **Test Script**: Created `check_orchestrator.py` (temporary) to verify:
    - Happy path (high confidence).
    - Low confidence warning.
    - Partial failure (no skeleton -> skip clinical).
- Result: All tests passed.

## Phase 3: VLM Integration

### 1. Interpretation Agent Update
- **Clinical Charts**: Updated `InterpretationAgent` to accept `clinical_charts` (Markdown tables).
- **Prompt Engineering**: Modified the LLM prompt to include these charts under a "## 상세 차트 데이터" section, giving the VLM direct access to quantitative data.
- **File**: `backend/services/interpretation_agent.py`

### 2. Report Agent Update
- **Data Flow**: Updated `ReportAgent` to extract `clinical_charts` from `AnalysisContext` and pass them to the `InterpretationAgent`.
- **Bug Fix**: Fixed `updrs_score` extraction logic (checking `total_score` first).
- **File**: `backend/agents/report_agent.py`

### 3. Verification
- **Test Script**: Created `check_vlm_prompt.py` to verify that the constructed prompt actually contains the chart data.
- **Result**: Verified successfully.

## Phase 4: Frontend Integration & API Cleanup

### 1. API Refactoring
- **Optimization**: Refactored `backend/routes/analyze.py` to remove redundant calls to `VisualizationService`.
- **Integration**: Updated the API to use the `heatmap_path` and `trajectory_map_path` generated by the `VisionAgent` and stored in `AnalysisContext`.
- **Response**: Ensured `visualization_urls` in the API response correctly points to these files.

### 2. Verification
- **Syntax Check**: Verified `analyze.py` syntax.
- **Manual Test**: User to perform end-to-end test via Frontend.
