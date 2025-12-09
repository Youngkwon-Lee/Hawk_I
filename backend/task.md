# Task: Multi-Agent Architecture & Feature Evolution

## Phase 1: Visual Feature Enhancement (Current)
- [x] **Implement Trajectory Map Generation** <!-- id: 0 -->
    - [x] Update `VisionAgent` to call `VisualizationService.generate_temporal_map` <!-- id: 1 -->
    - [x] Ensure map is generated for relevant tasks (Finger Tapping, Hand Movement) <!-- id: 2 -->
    - [x] Store path in `AnalysisContext` <!-- id: 3 -->
- [x] **Refine Heatmap Generation** <!-- id: 4 -->
# Task: Multi-Agent Architecture & Feature Evolution

## Phase 1: Visual Feature Enhancement (Current)
- [x] **Implement Trajectory Map Generation** <!-- id: 0 -->
    - [x] Update `VisionAgent` to call `VisualizationService.generate_temporal_map` <!-- id: 1 -->
    - [x] Ensure map is generated for relevant tasks (Finger Tapping, Hand Movement) <!-- id: 2 -->
    - [x] Store path in `AnalysisContext` <!-- id: 3 -->
- [x] **Refine Heatmap Generation** <!-- id: 4 -->
    - [x] Verify current heatmap generation in `VisionAgent` <!-- id: 5 -->
    - [x] Ensure it overlays correctly on video frames if needed <!-- id: 6 -->
- [ ] **Expose Visuals in API** <!-- id: 7 -->
    - [ ] Ensure `AnalysisContext` serializes these paths for the frontend <!-- id: 8 -->

## Phase 2: Collaborative Agent Structure
- [x] **Refactor Orchestrator** <!-- id: 9 -->
    - [x] Implement dynamic routing based on confidence <!-- id: 10 -->
- [x] **Enhance Clinical Agent** <!-- id: 11 -->
    - [x] Ensure it consumes Skeleton data to produce "Charts" (data structures for VLM) <!-- id: 12 -->

## Phase 3: VLM Integration (Interpretation Agent)
- [x] **Refactor Report Agent** <!-- id: 13 -->
    - [x] Update prompt to consume "Charts" and Visual Summaries <!-- id: 14 -->
    - [x] Ensure it generates the structured `Report` object <!-- id: 15 -->
    - [x] Construct prompt with Metrics + Visual descriptions <!-- id: 16 -->
    - [x] Generate natural language report <!-- id: 17 -->

## Phase 4: Frontend Integration & API Cleanup
- [x] **Refactor API** <!-- id: 18 -->
    - [x] Remove redundant visualization calls in `analyze.py` <!-- id: 19 -->
    - [x] Use `VisionAgent` outputs for response <!-- id: 20 -->
- [ ] **Verify Frontend** <!-- id: 21 -->
    - [ ] Manual end-to-end test <!-- id: 22 -->
