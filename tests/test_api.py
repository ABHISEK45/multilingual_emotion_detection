"""
Tests for the API module.

Tests all API endpoints, rate limiting, and utility functions.
"""

import pytest
import time
import datetime
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app
from src.api.utils import RateLimiter, format_timestamp, estimate_completion_time, cleanup_records
from src.api.models import TextRequest, BatchRequest, EmotionResponse, JobStatus

# Test data
SAMPLE_TEXTS = {
    'english': "I am feeling very happy today!",
    'hindi': "मुझे आज बहुत खुशी है।",
    'hinglish': "Main bahut happy feel kar raha hoon."
}

SAMPLE_BATCH = [
    "I am feeling very happy today!",
    "This news makes me so sad.",
    "मुझे आज बहुत खुशी है।",
    "Main bahut udaas hoon."
]

@pytest.fixture
def client():
    """TestClient fixture for making API requests."""
    return TestClient(app)

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    with patch("src.api.main.model") as mock:
        mock.predict_emotion.return_value = {"joy": 0.8, "sadness": 0.1, "anger": 0.05, "fear": 0.03, "surprise": 0.02}
        yield mock

@pytest.fixture
def mock_background_tasks():
    """Mock background tasks."""
    with patch("src.api.main.BackgroundTasks") as mock:
        yield mock

@pytest.fixture
def mock_job_store():
    """Mock job store for testing."""
    with patch("src.api.main.job_store") as mock:
        mock.return_value = {
            "job123": {
                "status": "completed",
                "total": 4,
                "processed": 4,
                "results": [
                    {"text": t, "emotions": {"joy": 0.8, "sadness": 0.1, "anger": 0.05, "fear": 0.03, "surprise": 0.02}}
                    for t in SAMPLE_BATCH
                ],
                "start_time": datetime.datetime.now() - datetime.timedelta(seconds=5),
                "estimated_completion": datetime.datetime.now() + datetime.timedelta(seconds=10),
            }
        }
        yield mock

# Test health check endpoint
def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime" in data

# Test single text analysis endpoint
def test_analyze_text(client, mock_model):
    """Test analyzing a single text."""
    for lang, text in SAMPLE_TEXTS.items():
        request = {"text": text, "language": lang if lang != "hinglish" else None}
        response = client.post("/analyze", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "emotions" in data
        assert "dominant_emotion" in data
        assert data["text"] == text
        assert isinstance(data["emotions"], dict)
        assert "joy" in data["emotions"]
        assert data["dominant_emotion"] in ["joy", "sadness", "anger", "fear", "surprise"]

# Test invalid input handling
def test_analyze_invalid_input(client):
    """Test handling of invalid input in analyze endpoint."""
    # Empty text
    response = client.post("/analyze", json={"text": "", "language": "en"})
    assert response.status_code == 422

    # Invalid language
    response = client.post("/analyze", json={"text": "Hello", "language": "invalid_lang"})
    assert response.status_code == 422

    # Missing text field
    response = client.post("/analyze", json={"language": "en"})
    assert response.status_code == 422

# Test batch processing endpoint
def test_batch_analyze_start(client, mock_model, mock_background_tasks):
    """Test starting a batch analysis job."""
    with patch("uuid.uuid4", return_value="job123"):
        request = {"texts": SAMPLE_BATCH}
        response = client.post("/batch/analyze", json=request)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["job_id"] == "job123"
        assert "status" in data
        assert data["status"] == "processing"

# Test batch job status endpoint
def test_get_job_status(client, mock_job_store):
    """Test getting the status of a batch job."""
    response = client.get("/batch/status/job123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job123"
    assert data["status"] == "completed"
    assert data["total"] == 4
    assert data["processed"] == 4
    assert "start_time" in data
    assert "estimated_completion" in data
    assert "results" in data
    assert len(data["results"]) == 4

# Test job not found
def test_job_not_found(client, mock_job_store):
    """Test response when job is not found."""
    response = client.get("/batch/status/nonexistent_job")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()

# Test rate limiting
def test_rate_limiter():
    """Test the rate limiter utility."""
    limiter = RateLimiter(limit=2, window=1)  # 2 requests per second
    
    # First two requests should pass
    assert limiter.is_allowed("test_client") is True
    assert limiter.is_allowed("test_client") is True
    
    # Third request should be rate limited
    assert limiter.is_allowed("test_client") is False
    
    # Different client should not be affected
    assert limiter.is_allowed("different_client") is True
    
    # Wait for reset
    time.sleep(1.1)
    
    # Should be allowed again
    assert limiter.is_allowed("test_client") is True

# Test middleware with rate limiting
def test_rate_limiting_middleware(client):
    """Test the rate limiting middleware."""
    # Mock the rate limiter to control its behavior
    with patch("src.api.main.rate_limiter") as mock_limiter:
        # First request allowed
        mock_limiter.is_allowed.return_value = True
        response = client.get("/health")
        assert response.status_code == 200
        
        # Second request rate limited
        mock_limiter.is_allowed.return_value = False
        response = client.get("/health")
        assert response.status_code == 429
        assert "retry" in response.json()["detail"].lower()

# Test utility functions
def test_format_timestamp():
    """Test timestamp formatting utility."""
    now = datetime.datetime.now()
    formatted = format_timestamp(now)
    assert isinstance(formatted, str)
    assert "T" in formatted  # ISO format has T between date and time
    assert "." in formatted  # Should include milliseconds

def test_estimate_completion_time():
    """Test completion time estimation utility."""
    start_time = datetime.datetime.now() - datetime.timedelta(seconds=10)
    total = 100
    processed = 25  # 25% complete after 10 seconds
    
    estimated = estimate_completion_time(start_time, processed, total)
    assert isinstance(estimated, datetime.datetime)
    
    # Should be approximately 30 seconds from start_time (40 seconds from now)
    expected_seconds = 30
    delta = abs((estimated - start_time).total_seconds() - expected_seconds)
    assert delta < 5  # Allow for small calculation differences

def test_cleanup_records():
    """Test job record cleanup utility."""
    # Create mock job store with old and new jobs
    now = datetime.datetime.now()
    two_hours_ago = now - datetime.timedelta(hours=2)
    
    job_store = {
        "recent_job": {
            "start_time": now - datetime.timedelta(minutes=15),
            "status": "completed"
        },
        "old_job": {
            "start_time": two_hours_ago,
            "status": "completed"
        },
        "old_error_job": {
            "start_time": two_hours_ago,
            "status": "error"
        }
    }
    
    # Clean up jobs older than 1 hour
    max_age = datetime.timedelta(hours=1)
    cleaned_store = cleanup_records(job_store, max_age)
    
    # Check that old jobs were removed
    assert "recent_job" in cleaned_store
    assert "old_job" not in cleaned_store
    assert "old_error_job" not in cleaned_store
    assert len(cleaned_store) == 1

# Test error handling
def test_model_error_handling(client, mock_model):
    """Test handling of model errors."""
    mock_model.predict_emotion.side_effect = Exception("Model error")
    
    response = client.post("/analyze", json={"text": "Hello", "language": "en"})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"].lower()

# Test documentation endpoints
def test_documentation_endpoints(client):
    """Test that documentation endpoints are available."""
    # OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    # Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    
    # ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200

