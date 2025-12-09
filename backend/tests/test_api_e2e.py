"""
E2E API Integration Tests for HawkEye Backend
Tests full API workflow: upload -> analyze -> result
"""

import pytest
import requests
import os
import time
from pathlib import Path

# Test configuration
BASE_URL = os.getenv("TEST_API_URL", "http://localhost:5000")
TEST_TIMEOUT = 120  # 2 minutes for analysis


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "HawkEye PD Backend"
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "ok"]


class TestAnalysisAPI:
    """Test video analysis API endpoints"""

    @pytest.fixture
    def sample_video_path(self):
        """Get path to sample test video"""
        # Check multiple possible locations
        possible_paths = [
            Path("tests/fixtures/sample_finger_tapping.mp4"),
            Path("../data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/15-000217_r.mp4"),
            Path("uploads/test_video.mp4"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        pytest.skip("No sample video available for E2E test")

    def test_analyze_without_file(self):
        """Test analysis endpoint rejects request without file"""
        response = requests.post(f"{BASE_URL}/api/analyze")
        assert response.status_code in [400, 422]

    def test_analyze_with_invalid_file(self):
        """Test analysis endpoint rejects non-video files"""
        # Create a fake text file
        files = {"video": ("test.txt", b"not a video", "text/plain")}
        response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        assert response.status_code in [400, 415, 422]

    @pytest.mark.slow
    def test_full_analysis_workflow(self, sample_video_path):
        """Test complete analysis workflow: upload -> analyze -> get result"""
        # Step 1: Upload and start analysis
        with open(sample_video_path, "rb") as video_file:
            files = {"video": (os.path.basename(sample_video_path), video_file, "video/mp4")}
            response = requests.post(f"{BASE_URL}/api/analysis/start", files=files)

        assert response.status_code == 200, f"Upload failed: {response.text}"
        data = response.json()
        assert "video_id" in data
        video_id = data["video_id"]

        # Step 2: Poll for progress
        start_time = time.time()
        while time.time() - start_time < TEST_TIMEOUT:
            progress_response = requests.get(f"{BASE_URL}/api/analysis/progress/{video_id}")
            if progress_response.status_code == 200:
                progress_data = progress_response.json()
                if progress_data.get("status") == "completed":
                    break
                if progress_data.get("status") == "error":
                    pytest.fail(f"Analysis failed: {progress_data.get('error')}")
            time.sleep(2)
        else:
            pytest.fail(f"Analysis timed out after {TEST_TIMEOUT} seconds")

        # Step 3: Get final result
        result_response = requests.get(f"{BASE_URL}/api/analysis/result/{video_id}")
        assert result_response.status_code == 200
        result = result_response.json()

        # Validate result structure
        assert "video_type" in result
        assert "metrics" in result
        assert "updrs_score" in result
        assert result["video_type"] in ["finger_tapping", "gait", "hand_movement", "leg_agility"]

    def test_get_nonexistent_result(self):
        """Test getting result for non-existent video ID"""
        response = requests.get(f"{BASE_URL}/api/analysis/result/nonexistent_id_12345")
        assert response.status_code == 404


class TestPopulationStats:
    """Test population statistics API"""

    def test_get_finger_tapping_stats(self):
        """Test getting finger tapping population statistics"""
        response = requests.get(f"{BASE_URL}/api/population-stats/finger_tapping")
        assert response.status_code == 200
        data = response.json()
        # API returns {"success": true, "data": {...}}
        assert data.get("success") is True
        assert "data" in data
        stats = data["data"]
        assert "score_distribution" in stats
        assert "metrics" in stats

    def test_get_gait_stats(self):
        """Test getting gait population statistics"""
        response = requests.get(f"{BASE_URL}/api/population-stats/gait")
        assert response.status_code == 200
        data = response.json()
        # API returns {"success": true, "data": {...}}
        assert data.get("success") is True
        assert "data" in data
        stats = data["data"]
        assert "score_distribution" in stats
        assert "metrics" in stats

    def test_invalid_task_type(self):
        """Test population stats for invalid task type"""
        response = requests.get(f"{BASE_URL}/api/population-stats/invalid_task")
        assert response.status_code in [400, 404]


class TestChatAPI:
    """Test chat/interpretation API"""

    def test_chat_without_context(self):
        """Test chat endpoint without analysis context"""
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": "What does UPDRS score 2 mean?"}
        )
        # Should work but may give generic response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_with_context(self):
        """Test chat endpoint with analysis context"""
        mock_context = {
            "video_type": "finger_tapping",
            "updrs_score": {"total_score": 2, "severity": "Mild"},
            "metrics": {
                "tapping_speed": 3.5,
                "amplitude_mean": 0.6,
                "fatigue_rate": 15.0
            }
        }
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "Explain this patient's results",
                "context": mock_context
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data


class TestHistoryAPI:
    """Test history API endpoints"""

    def test_get_history(self):
        """Test getting analysis history"""
        response = requests.get(f"{BASE_URL}/api/history/")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert "data" in data
        assert "items" in data["data"]
        assert "total" in data["data"]

    def test_get_history_with_filters(self):
        """Test getting history with filters"""
        params = {
            "task_type": "finger_tapping",
            "limit": 10,
            "sort": "date_desc"
        }
        response = requests.get(f"{BASE_URL}/api/history/", params=params)
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True

    def test_get_history_stats(self):
        """Test getting history statistics"""
        response = requests.get(f"{BASE_URL}/api/history/stats")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert "data" in data
        stats = data["data"]
        assert "total_analyses" in stats
        assert "task_distribution" in stats
        assert "score_distribution" in stats

    def test_get_nonexistent_history_item(self):
        """Test getting non-existent history item"""
        response = requests.get(f"{BASE_URL}/api/history/nonexistent_video_123")
        assert response.status_code == 404


class TestStreamingAPI:
    """Test streaming analysis API"""

    @pytest.fixture
    def sample_video_path(self):
        """Get path to sample test video"""
        possible_paths = [
            Path("tests/fixtures/sample_finger_tapping.mp4"),
            Path("uploads/test_video.mp4"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        pytest.skip("No sample video available for streaming test")

    @pytest.mark.slow
    def test_streaming_analysis(self, sample_video_path):
        """Test streaming analysis endpoint"""
        with open(sample_video_path, "rb") as video_file:
            files = {"video": (os.path.basename(sample_video_path), video_file, "video/mp4")}
            response = requests.post(
                f"{BASE_URL}/api/analysis/stream",
                files=files,
                stream=True
            )

        assert response.status_code == 200
        # Check that we receive streaming response
        content_type = response.headers.get("Content-Type", "")
        assert "text/event-stream" in content_type or "application/json" in content_type


# Pytest configuration
def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
