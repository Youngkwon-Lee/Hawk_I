# Implementation Plan - Phase 5: ML Model Integration

# Goal
Integrate the existing PyTorch LSTM models into the `ModelRegistry` system so that `ModelSelectorAgent` can dynamically choose between Rule-Based and ML models. Refactor `ClinicalAgent` to handle models that perform direct scoring (end-to-end) versus those that only extract metrics.

## Proposed Changes

### Backend

#### [MODIFY] [backend/models/lstm_pytorch_model.py](file:///c:/Users/YK/tulip/Hawkeye/backend/models/lstm_pytorch_model.py)
- Update `FingerTappingLSTMScorer` to detect and use CUDA (GPU) if available.
- Ensure model loading maps to the correct device.

#### [NEW] [backend/models/ml_wrapper.py](file:///c:/Users/YK/tulip/Hawkeye/backend/models/ml_wrapper.py)
- Create `FingerTappingMLModel` class implementing `AnalysisModel`.
- Wraps `FingerTappingLSTMScorer`.
- `process` method returns a dictionary containing both score and confidence, and potentially "simulated" metrics if needed, or a specific flag indicating direct scoring.

#### [MODIFY] [backend/agents/clinical_agent.py](file:///c:/Users/YK/tulip/Hawkeye/backend/agents/clinical_agent.py)
- Update `process` method logic:
    - Check if the output from `model_selector` contains a score.
    - If score exists, skip `UPDRSScorer` (or use it only for filling details).
    - If only metrics exist, run `UPDRSScorer`.

#### [MODIFY] [backend/app.py](file:///c:/Users/YK/tulip/Hawkeye/backend/app.py)
- Import and register `FingerTappingMLModel` on startup.

#### [MODIFY] [backend/services/model_registry.py](file:///c:/Users/YK/tulip/Hawkeye/backend/services/model_registry.py)
- (Optional) Add `is_scorer` flag to `ModelMetadata` to distinguish between feature extractors and end-to-end scorers.

## Verification Plan

### Automated Tests
- Run `backend/tests/test_agents.py` (if exists) or create a new test script to verify `ClinicalAgent` correctly handles both RuleBased and ML models.

### Manual Verification
1.  **Rule Based**: Upload a simple video, ensure RuleBased model is used (or force it by priority).
2.  **ML Model**: Set ML model priority higher, upload video, ensure LSTM scorer is used and result is displayed.
