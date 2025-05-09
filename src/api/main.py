"""
Main FastAPI application for the multilingual emotion detection system.
"""

import os
import time
import uuid
import uvicorn
import logging
import datetime
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional, Any, Callable
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary

from .endpoints import router, get_model, _model
from .models import ErrorResponse

)
logger = logging.getLogger(__name__)

# API statistics (in a production environment, use a database)
api_stats = {
    "start_time": None,
    "request_count": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_time": 0,
    "requests_by_endpoint": {},
    "errors_by_type": {},
}

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    'api_requests_total', 
    'Total count of API requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds', 
    'Request duration in seconds', 
    ['method', 'endpoint']
)

REQUESTS_IN_PROGRESS = Gauge(
    'api_requests_in_progress', 
    'Number of API requests in progress', 
    ['method', 'endpoint']
)

MODEL_PREDICTION_DURATION = Histogram(
    'api_model_prediction_duration_seconds', 
    'Model prediction duration in seconds'
)

EXCEPTIONS_TOTAL = Counter(
    'api_exceptions_total', 
    'Total count of API exceptions', 
    ['type']
)

# Create FastAPI app
app = FastAPI(
    title="Multilingual Emotion Detection API",
    description="""
    API for detecting emotions in multilingual text.
    
    Supports English and Hindi languages, with automatic language detection.
    """,
    version="1.0.0",
    docs_url=None,  # We'll configure a custom docs page
    redoc_url=None  # We'll configure a custom redoc page
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In a production environment, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api/v1")

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable):
    """Add a unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())
    # Add the request ID to the request state
    request.state.request_id = request_id
    
    # Measure request processing time
    start_time = time.time()
    
    # Get the endpoint path for statistics
    path = request.url.path
    method = request.method
    endpoint = f"{method} {path}"
    
    # Update request count
    api_stats["request_count"] += 1
    if endpoint not in api_stats["requests_by_endpoint"]:
        api_stats["requests_by_endpoint"][endpoint] = 0
    api_stats["requests_by_endpoint"][endpoint] += 1
    
    # Prometheus: Track in-progress requests
    REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()
    
    # Process the request
    try:
        # Prometheus: Track request duration
        with REQUEST_DURATION.labels(method=method, endpoint=path).time():
            response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Update successful request count
        status_code = response.status_code
        if 200 <= status_code < 400:
            api_stats["successful_requests"] += 1
        else:
            api_stats["failed_requests"] += 1
        
        # Prometheus: Count requests by status code
        REQUESTS_TOTAL.labels(method=method, endpoint=path, status_code=status_code).inc()
            
        # Log completion time for non-static requests
        if not path.startswith(("/docs", "/redoc", "/openapi.json", "/static")):
            elapsed = time.time() - start_time
            api_stats["total_processing_time"] += elapsed
            logger.info(f"Request {request_id} completed: {method} {path} - {status_code} in {elapsed:.3f}s")

        # Pr        return response
    except Exception as e:
        # Log the exception
        logger.error(f"Request {request_id} failed: {method} {path} - {str(e)}")
        api_stats["failed_requests"] += 1
        
        # Track error type
        error_type = type(e).__name__
        if error_type not in api_stats["errors_by_type"]:
            api_stats["errors_by_type"][error_type] = 0
        api_stats["errors_by_type"][error_type] += 1
        
        # Prometheus: Count exceptions by type
        EXCEPTIONS_TOTAL.labels(type=error_type).inc()
        REQUESTS_TOTAL.labels(method=method, endpoint=path, status_code=500).inc()
        
        # Prometheus: Decrement in-progress requests before re-raising
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()
        
        # Re-raise to let the exception handler deal with it
        raise

# Custom exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle exceptions and return structured error responses."""
    if isinstance(exc, HTTPException):
        # Handle HTTP exceptions with their specific status code
        status_code = exc.status_code
        detail = exc.detail
        headers = getattr(exc, "headers", None)
    else:
        # For non-HTTP exceptions, use 500 Internal Server Error
        status_code = 500
        detail = str(exc)
        headers = None
    
    # Get request ID if available
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Create error response
    error_response = ErrorResponse(
        detail=detail,
        error_type=type(exc).__name__,
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Log the error with request ID for tracing
    logger.error(f"Request {request_id} error: {detail}")
    
    # Create JSON response with appropriate status code and headers
    response = JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Add any additional headers from the exception
    if headers:
        for name, value in headers.items():
            response.headers[name] = value
    
    return response

# Root endpoint - redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root endpoint to documentation."""
    return RedirectResponse(url="/docs")

# Documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.1.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.1.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns status information about the API.
    """
    uptime = time.time() - api_stats["start_time"] if api_stats["start_time"] else 0
    
    return {
        "status": "healthy",
        "version": app.version,
        "uptime_seconds": uptime,
        "request_count": api_stats["request_count"],
        "success_rate": api_stats["successful_requests"] / max(1, api_stats["request_count"]),
        "model_loaded": _model is not None
    }

# Statistics endpoint
@app.get("/stats", response_model=Dict[str, Any], tags=["Monitoring"])
async def get_statistics():
    """
    Get API usage statistics.
    
    Returns detailed statistics about API usage.
    """
    uptime = time.time() - api_stats["start_time"] if api_stats["start_time"] else 0
    avg_response_time = api_stats["total_processing_time"] / max(1, api_stats["request_count"])
    
    return {
        "uptime_seconds": uptime,
        "requests": {
            "total": api_stats["request_count"],
            "successful": api_stats["successful_requests"],
            "failed": api_stats["failed_requests"],
            "success_rate": api_stats["successful_requests"] / max(1, api_stats["request_count"]),
        },
        "performance": {
            "average_response_time": avg_response_time,
            "total_processing_time": api_stats["total_processing_time"],
        },
        "endpoints": api_stats["requests_by_endpoint"],
        "errors": api_stats["errors_by_type"]
    }

# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Expose Prometheus metrics.
    
    This endpoint exposes metrics in the Prometheus format for monitoring.
    """
    return Response(
        content=prometheus_client.generate_latest(),
        media_type="text/plain"
    )

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting Multilingual Emotion Detection API")
    
    # Initialize statistics
    api_stats["start_time"] = time.time()
    api_stats["request_count"] = 0
    api_stats["successful_requests"] = 0
    api_stats["failed_requests"] = 0
    api_stats["total_processing_time"] = 0
    api_stats["requests_by_endpoint"] = {}
    api_stats["errors_by_type"] = {}
    
    # Preload the model if needed
    # Uncomment to preload the model at startup
    # _ = get_model()
    
    logger.info("API startup complete")

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down Multilingual Emotion Detection API")
    
    # Log final statistics
    uptime = time.time() - api_stats["start_time"] if api_stats["start_time"] else 0
    logger.info(f"API stats: uptime={uptime:.1f}s, requests={api_stats['request_count']}, "
               f"success_rate={api_stats['successful_requests'] / max(1, api_stats['request_count']):.2%}")
    
    # Clean up model resources if needed
    if _model is not None:
        logger.info("Cleaning up model resources")
        # Add any model cleanup here if needed
    
    logger.info("Shutdown complete")

