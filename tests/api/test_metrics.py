"""
Tests for the metrics module of the multilingual emotion detection API.

This module tests the Prometheus metrics collection, decorators, and utility
functions defined in src.api.metrics.
"""

import time
import asyncio
import prometheus_client
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from prometheus_client import Counter, Histogram, Gauge

from src.api.metrics import (
    # Constants and metrics
    REQUESTS_TOTAL, REQUEST_DURATION, REQUESTS_IN_PROGRESS, 
    MODEL_PREDICTION_DURATION, MODEL_ERRORS, EXCEPTIONS_TOTAL,
    MEMORY_USAGE, CPU_USAGE, THREAD_COUNT,
    
    # Utility functions
    increment_counter, observe_histogram, set_gauge, update_memory_metrics,
    
    # Decorators
    request_timing, async_request_timing, track_errors, 
    async_track_errors, model_timing, async_model_timing
)

# Reset registry before tests to avoid conflicts with previous test runs
prometheus_client.REGISTRY = prometheus_client.CollectorRegistry()

# Mock request object for testing decorators
class MockRequest:
    def __init__(self, method="GET", path="/test"):
        self.method = method
        self.url = MagicMock()
        self.url.path = path

# ===== Utility Function Tests =====

def test_increment_counter():
    """Test the increment_counter utility function."""
    test_counter = Counter('test_counter', 'Test counter', ['label1', 'label2'])
    
    # Increment counter with labels
    increment_counter(test_counter, label1="value1", label2="value2")
    
    # Check that counter was incremented
    assert test_counter.labels(label1="value1", label2="value2")._value.get() == 1
    
    # Increment again
    increment_counter(test_counter, label1="value1", label2="value2")
    assert test_counter.labels(label1="value1", label2="value2")._value.get() == 2
    
    # Different labels should have different values
    increment_counter(test_counter, label1="other", label2="value2")
    assert test_counter.labels(label1="other", label2="value2")._value.get() == 1

def test_observe_histogram():
    """Test the observe_histogram utility function."""
    test_histogram = Histogram('test_histogram', 'Test histogram', ['label1'])
    
    # Observe a value
    observe_histogram(test_histogram, 0.5, label1="value1")
    
    # Check histogram sum (should be equal to the value we observed)
    assert test_histogram.labels(label1="value1")._sum.get() == 0.5
    
    # Observe another value
    observe_histogram(test_histogram, 1.5, label1="value1")
    assert test_histogram.labels(label1="value1")._sum.get() == 2.0  # 0.5 + 1.5
    
    # Different labels should have different values
    observe_histogram(test_histogram, 2.0, label1="other")
    assert test_histogram.labels(label1="other")._sum.get() == 2.0

def test_set_gauge():
    """Test the set_gauge utility function."""
    test_gauge = Gauge('test_gauge', 'Test gauge', ['label1'])
    
    # Set gauge value
    set_gauge(test_gauge, 42, label1="value1")
    
    # Check gauge value
    assert test_gauge.labels(label1="value1")._value.get() == 42
    
    # Update gauge value
    set_gauge(test_gauge, 99, label1="value1")
    assert test_gauge.labels(label1="value1")._value.get() == 99
    
    # Different labels should have different values
    set_gauge(test_gauge, 123, label1="other")
    assert test_gauge.labels(label1="other")._value.get() == 123

@patch('psutil.Process')
def test_update_memory_metrics(mock_process):
    """Test the update_memory_metrics function."""
    # Setup mock process
    mock_process_instance = mock_process.return_value
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 1024 * 1024  # 1 MB
    mock_memory_info.vms = 2 * 1024 * 1024  # 2 MB
    mock_process_instance.memory_info.return_value = mock_memory_info
    mock_process_instance.cpu_percent.return_value = 25.5
    mock_process_instance.num_threads.return_value = 5
    
    # Call the function
    update_memory_metrics()
    
    # Check that gauges were set correctly
    assert MEMORY_USAGE.labels(type="rss")._value.get() == 1024 * 1024
    assert MEMORY_USAGE.labels(type="vms")._value.get() == 2 * 1024 * 1024
    assert CPU_USAGE._value.get() == 25.5
    assert THREAD_COUNT._value.get() == 5
    
    # Verify mocks were called
    mock_process_instance.memory_info.assert_called_once()
    mock_process_instance.cpu_percent.assert_called_once_with(interval=0.1)
    mock_process_instance.num_threads.assert_called_once()

# ===== Decorator Tests =====

def test_request_timing_decorator():
    """Test the request_timing decorator on a normal function."""
    mock_request = MockRequest(method="GET", path="/test")
    
    # Define a test function and decorate it
    @request_timing
    def test_function(request):
        return "success"
    
    # Call the function
    result = test_function(mock_request)
    assert result == "success"
    
    # Check that metrics were incremented
    assert REQUESTS_IN_PROGRESS.labels(method="GET", endpoint="/test")._value.get() == 0
    assert REQUESTS_TOTAL.labels(method="GET", endpoint="/test", status_code=200)._value.get() == 1
    
    # Histogram value should exist but we can't easily check the exact value
    # Just check that some observation was made
    assert REQUEST_DURATION.labels(method="GET", endpoint="/test")._count.get() == 1

def test_request_timing_decorator_with_exception():
    """Test the request_timing decorator when function raises an exception."""
    mock_request = MockRequest(method="POST", path="/api/resource")
    
    # Define a test function that raises an exception
    @request_timing
    def failing_function(request):
        raise ValueError("Test error")
    
    # Call the function and expect exception
    with pytest.raises(ValueError):
        failing_function(mock_request)
    
    # Check that metrics were incremented
    assert REQUESTS_IN_PROGRESS.labels(method="POST", endpoint="/api/resource")._value.get() == 0
    assert REQUESTS_TOTAL.labels(method="POST", endpoint="/api/resource", status_code=500)._value.get() == 1
    assert EXCEPTIONS_TOTAL.labels(type="ValueError", endpoint="/api/resource")._value.get() == 1
    assert REQUEST_DURATION.labels(method="POST", endpoint="/api/resource")._count.get() == 1

def test_request_timing_without_request_object():
    """Test request_timing decorator with arguments that don't include a request."""
    @request_timing
    def function_without_request(arg1, arg2):
        return arg1 + arg2
    
    result = function_without_request(1, 2)
    assert result == 3
    
    # Should use UNKNOWN for method and endpoint
    assert REQUESTS_TOTAL.labels(method="UNKNOWN", endpoint="UNKNOWN", status_code=200)._value.get() == 1

@pytest.mark.asyncio
async def test_async_request_timing_decorator():
    """Test the async_request_timing decorator."""
    mock_request = MockRequest(method="GET", path="/async_test")
    
    # Define a test async function and decorate it
    @async_request_timing
    async def test_async_function(request):
        await asyncio.sleep(0.01)  # Small sleep to simulate async work
        return "async success"
    
    # Call the function
    result = await test_async_function(mock_request)
    assert result == "async success"
    
    # Check that metrics were incremented
    assert REQUESTS_IN_PROGRESS.labels(method="GET", endpoint="/async_test")._value.get() == 0
    assert REQUESTS_TOTAL.labels(method="GET", endpoint="/async_test", status_code=200)._value.get() == 1
    assert REQUEST_DURATION.labels(method="GET", endpoint="/async_test")._count.get() == 1

def test_track_errors_decorator():
    """Test the track_errors decorator."""
    # Define a test function that doesn't raise
    @track_errors
    def successful_function(arg):
        return arg * 2
    
    # Call the function
    result = successful_function(21)
    assert result == 42
    
    # Define a function that raises an exception
    @track_errors
    def failing_function():
        raise RuntimeError("Test error")
    
    # Call the function and expect exception
    with pytest.raises(RuntimeError):
        failing_function()
    
    # Check that exception was counted
    assert EXCEPTIONS_TOTAL.labels(type="RuntimeError", endpoint="function")._value.get() == 1

@pytest.mark.asyncio
async def test_async_track_errors_decorator():
    """Test the async_track_errors decorator."""
    # Define a test async function that doesn't raise
    @async_track_errors
    async def successful_async_function(arg):
        await asyncio.sleep(0.01)
        return arg * 2
    
    # Call the function
    result = await successful_async_function(21)
    assert result == 42
    
    # Define an async function that raises an exception
    @async_track_errors
    async def failing_async_function():
        await asyncio.sleep(0.01)
        raise RuntimeError("Test async error")
    
    # Call the function and expect exception
    with pytest.raises(RuntimeError):
        await failing_async_function()
    
    # Check that exception was counted
    assert EXCEPTIONS_TOTAL.labels(type="RuntimeError", endpoint="function")._value.get() == 1

def test_model_timing_decorator():
    """Test the model_timing decorator."""
    # Define a test model predict function
    @model_timing
    def model_predict(self, texts):
        time.sleep(0.01)  # Small delay to simulate processing
        return [{"result": t} for t in texts]
    
    # Create a dummy class instance
    class DummyModel:
        pass
    
    model = DummyModel()
    
    # Test with a list of texts
    texts = ["text1", "text2", "text3"]
    results = model_predict(model, texts)
    
    # Check results
    assert len(results) == 3
    assert results[0]["result"] == "text1"
    
    # Check that metrics were recorded
    assert MODEL_PREDICTION_DURATION.labels(batch_size="3")._count.get() == 1
    
    # Test with kwargs
    results = model_predict(model, texts=["single text"])
    assert len(results) == 1
    assert MODEL_PREDICTION_DURATION.labels(batch_size="1")._count.get() == 1

def test_model_timing_with_error():
    """Test the model_timing decorator when model raises an error."""
    @model_timing
    def failing_model_predict(self, batch):
        raise ValueError("Model error")
    
    # Create a dummy instance
    model = MagicMock()
    
    # Call the function and expect exception
    with pytest.raises(ValueError):
        failing_model_predict(model, ["text"])
    
    # Check that error was counted
    assert MODEL_ERRORS.labels(error_type="ValueError")._value.get() == 1

@pytest.mark.asyncio
async def test_async_model_timing_decorator():
    """Test the async_model_timing decorator."""
    # Define a test async model predict function
    @async_model_timing
    async def async_model_predict(self, texts):
        await asyncio.sleep(0.01)  # Small delay to simulate processing
        return [{"result": t} for t in texts]
    
    # Create a dummy class instance
    class DummyAsyncModel:
        pass
    
    model = DummyAsyncModel()
    
    # Test with a list of texts
    texts = ["text1", "text2"]
    results = await async_model_predict(model, texts)
    
    # Check results
    assert len(results) == 2
    assert results[0]["result"] == "text1"
    
    # Check that metrics were recorded
    assert MODEL_PREDICTION_DURATION.labels(batch_size="2")._count.get() == 1
    
    # Test with other patterns of arguments
    results = await async_model_predict(model, batch=["single text"])
    assert len(results) == 1
    assert MODEL_PREDICTION_DURATION.labels(batch_size="1")._count.get() == 1

# ===== Integration Tests =====

def test_metrics_in_fastapi_request_context():
    """Test metrics collection in a simulated FastAPI request context."""
    # This test would typically be an integration test with FastAPI
    # For unit tests, we can simulate the request context
    mock_request = MockRequest(method="GET", path="/api/test")
    
    @request_timing
    def api_endpoint(request):
        # Simulate API processing
        time.sleep(0.01)
        
        # Update memory metrics
        update_memory_metrics()
        
        # Use model
        @model_timing
        def model_function(texts):
            return [{"score": 0.9} for _ in texts]
            
        model_function(["sample text"])
        
        return {"status": "success"}
    
    # Call the function
    result = api_endpoint(mock_request)
    assert result["status"] == "success"
    
    # Check various metrics were updated
    assert REQUESTS_TOTAL.labels(method="GET", endpoint="/api/test", status_code=200)._value.get() == 1
    assert REQUEST_DURATION.labels(method="GET", endpoint="/api/test")._count.get() == 1
    assert MODEL_PREDICTION_DURATION.labels(batch_size="1")._count.get() == 1
    # Memory metrics should also be set but values will vary
