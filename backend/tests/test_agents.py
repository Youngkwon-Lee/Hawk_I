"""
Unit Tests for Hawkeye Backend Agents
"""
import pytest


class TestClinicalAgent:
    """Tests for Clinical Agent"""

    def test_clinical_agent_process_finger_tapping(self, mock_finger_tapping_landmarks, analysis_context):
        """Test ClinicalAgent processes finger tapping data correctly"""
        from agents.clinical_agent import ClinicalAgent

        agent = ClinicalAgent()

        # Setup context
        analysis_context.task_type = "finger_tapping"
        analysis_context.status = "vision_done"
        analysis_context.vision_meta = {"fps": 30.0}
        analysis_context.skeleton_data = {"landmarks": mock_finger_tapping_landmarks}

        # Process
        result = agent.process(analysis_context)

        assert result.error is None, f"Should not have error: {result.error}"
        assert result.kinematic_metrics is not None, "Should have kinematic metrics"
        assert result.clinical_scores is not None, "Should have clinical scores"
        assert "total_score" in result.clinical_scores

    def test_clinical_agent_generates_charts(self, mock_finger_tapping_landmarks, analysis_context):
        """Test ClinicalAgent generates clinical charts"""
        from agents.clinical_agent import ClinicalAgent

        agent = ClinicalAgent()

        analysis_context.task_type = "finger_tapping"
        analysis_context.status = "vision_done"
        analysis_context.vision_meta = {"fps": 30.0}
        analysis_context.skeleton_data = {"landmarks": mock_finger_tapping_landmarks}

        result = agent.process(analysis_context)

        assert result.clinical_charts is not None, "Should generate clinical charts"
        assert isinstance(result.clinical_charts, str)


class TestModelSelectorAgent:
    """Tests for Model Selector Agent"""

    def test_model_selector_initialization(self):
        """Test ModelSelectorAgent initializes correctly"""
        from agents.model_selector_agent import ModelSelectorAgent

        agent = ModelSelectorAgent()
        assert agent is not None

    def test_model_selector_process(self, mock_finger_tapping_landmarks, analysis_context):
        """Test ModelSelectorAgent processes context correctly"""
        from agents.model_selector_agent import ModelSelectorAgent
        from agents.clinical_agent import ClinicalAgent

        # First run clinical agent to populate metrics
        clinical = ClinicalAgent()
        analysis_context.task_type = "finger_tapping"
        analysis_context.status = "vision_done"
        analysis_context.vision_meta = {"fps": 30.0}
        analysis_context.skeleton_data = {"landmarks": mock_finger_tapping_landmarks}
        analysis_context = clinical.process(analysis_context)

        # Model selector processes context
        selector = ModelSelectorAgent()
        result = selector.process(analysis_context)

        # Should have clinical scores after processing
        assert result.clinical_scores is not None


class TestValidationAgent:
    """Tests for Validation Agent"""

    def test_validation_agent_validates_context(self, mock_finger_tapping_landmarks, analysis_context):
        """Test ValidationAgent validates analysis results"""
        from agents.validation_agent import ValidationAgent
        from agents.clinical_agent import ClinicalAgent

        # First process with clinical agent
        clinical = ClinicalAgent()
        analysis_context.task_type = "finger_tapping"
        analysis_context.status = "vision_done"
        analysis_context.vision_meta = {"fps": 30.0}
        analysis_context.skeleton_data = {"landmarks": mock_finger_tapping_landmarks}
        analysis_context = clinical.process(analysis_context)

        # Then validate
        validator = ValidationAgent()
        result = validator.process(analysis_context)

        assert result.validation_result is not None, "Should have validation result"
        assert "is_valid" in result.validation_result or "quality_score" in result.validation_result


class TestOrchestrator:
    """Tests for Orchestrator Agent"""

    @pytest.mark.skip(reason="Requires mediapipe installation")
    def test_orchestrator_initialization(self):
        """Test Orchestrator initializes correctly"""
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        assert orchestrator is not None

    @pytest.mark.skip(reason="Requires mediapipe and video file for full E2E")
    def test_orchestrator_finger_tapping_flow(self, mock_finger_tapping_landmarks, analysis_context, mocker):
        """Test Orchestrator runs full finger tapping analysis flow"""
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        assert orchestrator is not None


class TestReportAgent:
    """Tests for Report Agent"""

    def test_report_agent_initialization(self):
        """Test ReportAgent initializes correctly"""
        from agents.report_agent import ReportAgent

        agent = ReportAgent()
        assert agent is not None

    def test_report_agent_without_openai(self, mock_finger_tapping_landmarks, analysis_context, mocker):
        """Test ReportAgent generates fallback report without OpenAI"""
        from agents.report_agent import ReportAgent
        from agents.clinical_agent import ClinicalAgent

        # Process clinical first
        clinical = ClinicalAgent()
        analysis_context.task_type = "finger_tapping"
        analysis_context.status = "vision_done"
        analysis_context.vision_meta = {"fps": 30.0}
        analysis_context.skeleton_data = {"landmarks": mock_finger_tapping_landmarks}
        analysis_context = clinical.process(analysis_context)

        # Mock OpenAI to fail
        mocker.patch.dict('os.environ', {'OPENAI_API_KEY': ''})

        # Report agent should still work with fallback
        reporter = ReportAgent()
        result = reporter.process(analysis_context)

        # Should either succeed with AI or fall back gracefully
        assert result.error is None or "report" in str(result.error).lower()
