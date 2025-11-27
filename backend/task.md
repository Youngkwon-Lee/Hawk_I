# Task: Multi-Agent Architecture & Feature Evolution

## Phase 1: Visual Feature Enhancement (Current)
- [ ] **Implement Trajectory Map Generation** <!-- id: 0 -->
    - [ ] Update `VisionAgent` to call `VisualizationService.generate_temporal_map` <!-- id: 1 -->
    - [ ] Ensure map is generated for relevant tasks (Finger Tapping, Hand Movement) <!-- id: 2 -->
    - [ ] Store path in `AnalysisContext` <!-- id: 3 -->
- [ ] **Refine Heatmap Generation** <!-- id: 4 -->
    - [ ] Verify current heatmap generation in `VisionAgent` <!-- id: 5 -->
    - [ ] Ensure it overlays correctly on video frames if needed <!-- id: 6 -->
- [ ] **Expose Visuals in API** <!-- id: 7 -->
    - [ ] Ensure `AnalysisContext` serializes these paths for the frontend <!-- id: 8 -->

## Phase 2: Collaborative Agent Structure
- [ ] **Refactor Orchestrator** <!-- id: 9 -->
    - [ ] Implement dynamic routing based on confidence <!-- id: 10 -->
- [ ] **Enhance Clinical Agent** <!-- id: 11 -->
    - [ ] Ensure it consumes Skeleton data to produce "Charts" (data structures for VLM) <!-- id: 12 -->

## Phase 3: VLM Integration (Interpretation Agent)
- [ ] **Integrate VLM Client** <!-- id: 13 -->
    - [ ] Setup OpenAI/Gemini client <!-- id: 14 -->
- [ ] **Implement Interpretation Logic** <!-- id: 15 -->
    - [ ] Construct prompt with Metrics + Visual descriptions <!-- id: 16 -->
    - [ ] Generate natural language report <!-- id: 17 -->
