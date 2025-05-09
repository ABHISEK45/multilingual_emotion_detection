"""
Centralized metrics module for the multilingual emotion detection system.

This module provides Prometheus metric collectors for tracking API usage,
performance, and errors. It includes decorators and utility functions to
simplify metrics collection throughout the application.

Usage:
    from src.api.metrics import request_timing, track_errors, increment_counter
    
    @request_timing
    def my_function():
        # Function execution time will be tracked
        pass
        
    @track_errors
    def error_prone_function():
        # Errors will be counted by type
        pass
        
    # Increment a counter manually
    increment_counter(REQUESTS_TOTAL, method="GET", endpoint="/api/resource")
"""

import time
import psutil
import functools
import traceback
import logging
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic function decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# =============================================================================
# API Request Metrics
# =============================================================================

# Total count of API requests
REQUESTS_TOTAL = Counter(
    'api_requests_total', 
    'Total count of API requests', 
    ['method', 'endpoint', 'status_code']
)

# Request duration in seconds
REQUEST_DURATION = Histogram(
    'api_request_duration_seconds', 
    'Request duration in seconds', 
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Number of active/in-progress requests
REQUESTS_IN_PROGRESS = Gauge(
    'api_requests_in_progress', 
    'Number of API requests in progress', 
    ['method', 'endpoint']
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    'api_rate_limit_exceeded_total', 
    'Total count of rate limit exceeded',
    ['endpoint', 'client_id']
)

RATE_LIMIT_REMAINING = Gauge(
    'api_rate_limit_remaining', 
    'Remaining rate limit for clients',
    ['client_id']
)

# =============================================================================
# Model Inference Metrics
# =============================================================================

# Model prediction duration in seconds
MODEL_PREDICTION_DURATION = Histogram(
    'api_model_prediction_duration_seconds', 
    'Model prediction duration in seconds',
    ['batch_size'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Model errors
MODEL_ERRORS = Counter(
    'api_model_errors_total', 
    'Total count of model prediction errors',
    ['error_type']
)

# Model cache hits/misses
MODEL_CACHE_HITS = Counter(
    'api_model_cache_hits_total',
    'Total count of model cache hits'
)

MODEL_CACHE_MISSES = Counter(
    'api_model_cache_misses_total',
    'Total count of model cache misses'
)

# =============================================================================
# Job & Batch Processing Metrics
# =============================================================================

# Active jobs by status
ACTIVE_JOBS = Gauge(
    'api_jobs_active', 
    'Number of active background jobs',
    ['status']  # pending, processing, completed, failed
)

# Total number of texts processed in batch jobs
JOB_TEXTS_TOTAL = Counter(
    'api_job_texts_total', 
    'Total number of texts processed in batch jobs',
    ['status']  # success, error
)

# Job processing time in seconds
JOB_PROCESSING_TIME = Histogram(
    'api_job_processing_seconds', 
    'Job processing time in seconds',
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

# Average number of texts processed per second in a job
JOB_TEXTS_PER_SECOND = Gauge(
    'api_job_texts_per_second', 
    'Average number of texts processed per second in a job'
)

# =============================================================================
# Language Detection Metrics
# =============================================================================

# Language detection counts
LANGUAGE_DETECTION_COUNT = Counter(
    'api_language_detection_total', 
    'Total count of language detections',
    ['detected_language']
)

# =============================================================================
# Error Metrics
# =============================================================================

# Exception counts by type
EXCEPTIONS_TOTAL = Counter(
    'api_exceptions_total', 
    'Total count of API exceptions', 
    ['type', 'endpoint']
)

# =============================================================================
# System Metrics
# =============================================================================

# Memory usage
MEMORY_USAGE = Gauge(
    'api_memory_usage_bytes',
    'Memory usage of the API process in bytes',
    ['type']  # rss, vms
)

# CPU usage percentage
CPU_USAGE = Gauge(
    'api_cpu_usage_percent',
    'CPU usage percentage of the API process'
)

# Thread count
THREAD_COUNT = Gauge(
    'api_thread_count',
    'Number of threads in the API process'
)

# =============================================================================
# Utility Functions
# =============================================================================

def increment_counter(counter: Counter, **labels) -> None:
    """
    Increment a Prometheus counter with the given labels.
    
    Args:
        counter: Prometheus Counter object
        **labels: Label values to apply
    """
    counter.labels(**labels).inc()

def observe_histogram(histogram: Histogram, value: float, **labels) -> None:
    """
    Record a value in a Prometheus histogram with the given labels.
    
    Args:
        histogram: Prometheus Histogram object
        value: Value to record
        **labels: Label values to apply
    """
    histogram.labels(**labels).observe(value)

def set_gauge(gauge: Gauge, value: float, **labels) -> None:
    """
    Set a Prometheus gauge to the given value with the given labels.
    
    Args:
        gauge: Prometheus Gauge object
        value: Value to set
        **labels: Label values to apply
    """
    gauge.labels(**labels).set(value)

def update_memory_metrics() -> None:
    """
    Update memory usage metrics for the current process.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Record resident set size (RSS) - actual memory used
    MEMORY_USAGE.labels(type="rss").set(memory_info.rss)
    
    # Record virtual memory size (VMS)
    MEMORY_USAGE.labels(type="vms").set(memory_info.vms)
    
    # Record CPU usage
    CPU_USAGE.set(process.cpu_percent(interval=0.1))
    
    # Record thread count
    THREAD_COUNT.set(process.num_threads())

# =============================================================================
# Decorator Functions
# =============================================================================

def request_timing(func: F) -> F:
    """
    Decorator to measure and record the execution time of a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Decorated function that reports timing metrics
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract method and endpoint from the request object
        # This assumes the first arg is a FastAPI Request or similar object
        method = "UNKNOWN"
        endpoint = "UNKNOWN"
        
        if args and hasattr(args[0], "method") and hasattr(args[0], "url"):
            request = args[0]
            method = request.method
            endpoint = request.url.path
        
        # Increment in-progress counter
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        status_code = 200  # Default success
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Track exception and re-raise
            status_code = getattr(e, "status_code", 500)
            EXCEPTIONS_TOTAL.labels(type=type(e).__name__, endpoint=endpoint).inc()
            raise
        finally:
            # Record metrics regardless of success/failure
            duration = time.time() - start_time
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
    
    return wrapper

def async_request_timing(func: AsyncF) -> AsyncF:
    """
    Decorator to measure and record the execution time of an async function.
    
    Args:
        func: The async function to be decorated
        
    Returns:
        Decorated async function that reports timing metrics
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract method and endpoint from the request object
        # This assumes the first arg is a FastAPI Request or similar object
        method = "UNKNOWN"
        endpoint = "UNKNOWN"
        
        if args and hasattr(args[0], "method") and hasattr(args[0], "url"):
            request = args[0]
            method = request.method
            endpoint = request.url.path
        
        # Increment in-progress counter
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        status_code = 200  # Default success
        
        try:
            # Execute the async function
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            # Track exception and re-raise
            status_code = getattr(e, "status_code", 500)
            EXCEPTIONS_TOTAL.labels(type=type(e).__name__, endpoint=endpoint).inc()
            raise
        finally:
            # Record metrics regardless of success/failure
            duration = time.time() - start_time
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
    
    return wrapper

def track_errors(func: F) -> F:
    """
    Decorator to track exceptions raised by a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Decorated function that reports error metrics
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Increment exception counter
            EXCEPTIONS_TOTAL.labels(type=type(e).__name__, endpoint="function").inc()
            # Log the error
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Re-raise the exception
            raise
    
    return wrapper

def async_track_errors(func: AsyncF) -> AsyncF:
    """
    Decorator to track exceptions raised by an async function.
    
    Args:
        func: The async function to be decorated
        
    Returns:
        Decorated async function that reports error metrics
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Increment exception counter
            EXCEPTIONS_TOTAL.labels(type=type(e).__name__, endpoint="function").inc()
            # Log the error
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Re-raise the exception
            raise
    
    return wrapper

def model_timing(func: F) -> F:
    """
    Decorator to measure and record the execution time of model inference.
    
    Args:
        func: The function to be decorated (typically a model's predict method)
        
    Returns:
        Decorated function that reports model timing metrics
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Determine batch size - attempt to extract from args or kwargs
        batch_size = "1"  # Default for single item
        
        # Try to extract batch size from first argument after self
        if len(args) > 1 and hasattr(args[1], "__len__"):
            batch_size = str(len(args[1]))
        elif "batch" in kwargs and hasattr(kwargs["batch"], "__len__"):
            batch_size = str(len(kwargs["batch"]))
        elif "texts" in kwargs and hasattr(kwargs["texts"], "__len__"):
            batch_size = str(len(kwargs["texts"]))
        
        start_time = time.time()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Record model prediction time
            duration = time.time() - start_time
            MODEL_PREDICTION_DURATION.labels(batch_size=batch_size).observe(duration)
            
            return result
        except Exception as e:
            # Track model errors
            MODEL_ERRORS.labels(error_type=type(e).__name__).inc()
            # Re-raise the exception
            raise
    
    return wrapper

def async_model_timing(func: AsyncF) -> AsyncF:
    """
    Decorator to measure and record the execution time of async model inference.
    
    Args:
        func: The async function to be decorated
        
    Returns:
        Decorated async function that reports model timing metrics
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Determine batch size - attempt to extract from args or kwargs
        batch_size = "1"  # Default for single item
        
        # Try to extract batch size from first argument after self
        if len(args) > 1 and hasattr(args[1], "__len__"):
            batch_size = str(len(args[1]))
        elif "batch" in kwargs and hasattr(kwargs["batch"], "__len__"):
            batch_size = str(len(kwargs["batch"]))
        elif "texts" in kwargs and hasattr(kwargs["texts"], "__len__"):
            batch_size = str(len(kwargs["texts"]))
        
        start_time = time.time()
        
        try:
            # Execute the async function
            result = await func(*args, **kwargs)
            
            # Record model prediction time
            duration = time.time() - start_time
            MODEL_PREDICTION_DURATION.labels(batch_size=batch_size).observe(duration)
            
            return result
        except Exception as e:
            # Track model errors
            MODEL_ERRORS.labels(error_type=type(e).__name__).inc()
            # Re-raise the exception
            raise
    
    return wrapper
