"""
API endpoints for the multilingual emotion detection system.
"""

import uuid
import time
import logging
import asyncio
import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Response, Depends, Header
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Tuple
import traceback
from prometheus_client import Counter, Histogram, Gauge, Summary

from .models import (
    TextRequest, BatchRequest, EmotionResponse, 
    BatchJobResponse, JobStatus, ErrorResponse, SupportedLanguage
)
from .utils import RateLimiter
from src.model import EmotionDetectionModel
from src.preprocessor import preprocess_text, detect_language, validate_text

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory job store
# In a production environment, this would be replaced by a database
jobs: Dict[str, Dict[str, Any]] = {}

# Rate limiter
rate_limiter = RateLimiter(limit=100, window=60)  # 100 requests per minute

# Lazy-loaded model singleton
_model = None

# Prometheus metrics
# Request metrics
REQUEST_COUNT = Counter(
    'emotion_api_requests_total', 
    'Total count of API requests',
    ['endpoint', 'method', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'emotion_api_request_duration_seconds', 
    'Request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    'emotion_api_rate_limit_exceeded_total', 
    'Total count of rate limit exceeded',
    ['endpoint', 'client_id']
)
RATE_LIMIT_REMAINING = Gauge(
    'emotion_api_rate_limit_remaining', 
    'Remaining rate limit for clients',
    ['client_id']
)

# Job metrics
ACTIVE_JOBS = Gauge(
    'emotion_api_jobs_active', 
    'Number of active background jobs',
    ['status']  # pending, processing, completed, failed
)
JOB_TEXTS_TOTAL = Counter(
    'emotion_api_job_texts_total', 
    'Total number of texts processed in batch jobs'
)
JOB_PROCESSING_TIME = Histogram(
    'emotion_api_job_processing_seconds', 
    'Job processing time in seconds',
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)
JOB_TEXTS_PER_SECOND = Gauge(
    'emotion_api_job_texts_per_second', 
    'Average number of texts processed per second in a job'
)

# Model metrics
MODEL_PREDICTION_DURATION = Histogram(
    'emotion_api_model_prediction_seconds', 
    'Model prediction time in seconds',
    ['batch_size'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)
MODEL_ERRORS = Counter(
    'emotion_api_model_errors_total', 
    'Total count of model prediction errors'
)

# Language detection metrics
LANGUAGE_DETECTION_COUNT = Counter(
    'emotion_api_language_detection_total', 
    'Total count of language detections',
    ['detected_language']
)

def get_model():
    """Get or initialize the model singleton."""
    global _model
    if _model is None:
        logger.info("Initializing emotion detection model")
        _model = EmotionDetectionModel()
    return _model


def get_client_id(request: Request) -> str:
    """Extract client ID from request for rate limiting."""
    # In a production environment, you might use more sophisticated methods
    # such as API keys or authenticated user IDs
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_id = forwarded.split(",")[0].strip()
    else:
        client_id = request.client.host if request.client else "unknown"
    return client_id


def check_rate_limit(request: Request) -> Tuple[bool, Dict[str, Any]]:
    """Check if request is allowed based on rate limiting."""
    client_id = get_client_id(request)
    allowed = rate_limiter.is_allowed(client_id)
    headers = {
        "X-RateLimit-Limit": str(rate_limiter.limit),
        "X-RateLimit-Remaining": str(rate_limiter.get_remaining(client_id)),
        "X-RateLimit-Reset": str(int(rate_limiter.get_reset_time(client_id)))
    }
    return allowed, headers


@router.post("/detect", response_model=EmotionResponse, responses={
    200: {"description": "Successful emotion detection"},
    400: {"model": ErrorResponse, "description": "Invalid input"},
    429: {"model": ErrorResponse, "description": "Too many requests"},
    500: {"model": ErrorResponse, "description": "Server error during processing"}
})
async def detect_emotion(request: TextRequest, req: Request):
    """Detect emotions in a single text."""
    endpoint = "/detect"
    client_id = get_client_id(req)
    start_time = time.time()
    status_code = 200
    
    try:
        # Check rate limit
        allowed, rate_limit_headers = check_rate_limit(req)
        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            RATE_LIMIT_EXCEEDED.labels(endpoint=endpoint, client_id=client_id).inc()
            status_code = 429
            response = JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    detail="Rate limit exceeded. Try again later.",
                    error_type="RateLimitExceeded",
                    timestamp=datetime.datetime.now().isoformat()
                ).dict()
            )
            # Add rate limit headers
            for k, v in rate_limit_headers.items():
                response.headers[k] = v
                
            # Record metrics
            request_duration = time.time() - start_time
            REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status_code=status_code).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(request_duration)
            RATE_LIMIT_REMAINING.labels(client_id=client_id).set(rate_limiter.get_remaining(client_id))
                
            return response
        
        model = get_model()
        detected_language = None
        
        try:
            # Validate input text
            is_valid, error_message = validate_text(request.text)
            if not is_valid:
                raise ValueError(f"Invalid input text: {error_message}")
            
            # Auto-detect language if needed
            lang = request.language.value  # Convert from enum to string
            if lang == "auto":
                detected_language = detect_language(request.text)
                lang = detected_language
                logger.debug(f"Detected language: {lang}")
                LANGUAGE_DETECTION_COUNT.labels(detected_language=lang).inc()
            
            # Preprocess text
            logger.debug(f"Preprocessing text: {request.text[:50]}...")
            processed_text = preprocess_text(request.text, lang)
            
            # Get prediction
            logger.debug("Predicting emotions")
            # Measure model prediction time with Prometheus
            with MODEL_PREDICTION_DURATION.labels(batch_size="1").time():
                result = model.predict(processed_text)
            
            # Add additional information
            processing_time_ms = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time_ms
            
            if detected_language:
                result["language_detected"] = detected_language
                
            logger.info(f"Processed text in {processing_time_ms:.2f}ms, dominant emotion: {result['dominant_emotion']}")
            
            # Prepare response with rate limit headers
            response = JSONResponse(
                status_code=200,
                content=result
            )
            
            # Add rate limit headers
            for k, v in rate_limit_headers.items():
                response.headers[k] = v
                
            # Record metrics
            RATE_LIMIT_REMAINING.labels(client_id=client_id).set(rate_limiter.get_remaining(client_id))
            request_duration = time.time() - start_time
            REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status_code=200).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(request_duration)
                
            return response
            
        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}")
            status_code = 400
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            logger.debug(traceback.format_exc())
            MODEL_ERRORS.inc()
            status_code = 500
            raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
            
    except HTTPException as e:
        # Record metrics for error responses
        request_duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status_code=e.status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(request_duration)
        raise
    except Exception as e:
        # Catch any other exceptions
        request_duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status_code=500).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(request_duration)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/batch-predict", response_model=List[Dict[str, Any]], responses={
    200: {"description": "Successful batch emotion detection"},
    400: {"model": ErrorResponse, "description": "Invalid input"},
    429: {"model": ErrorResponse, "description": "Too many requests"},
    500: {"model": ErrorResponse, "description": "Server error during processing"}
})
async def batch_predict_emotion(batch_request: BatchRequest, req: Request):
    """Detect emotions in a batch of texts."""
    endpoint = "/batch-predict"
    client_id = get_client_id(req)
    start_time = time.time()
    status_code = 200
    
    try:
        # Check rate limit
        allowed, rate_limit_headers = check_rate_limit(req)
        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            RATE_LIMIT_EXCEEDED.labels(endpoint=endpoint, client_id=client_id).inc()
            status_code = 429
            response = JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    detail="Rate limit exceeded. Try again later.",
                    error_type="RateLimitExceeded",
                    timestamp=datetime.datetime.now().isoformat()
                ).dict()
            )
            # Add rate limit headers
            for k, v in rate_limit_headers.items():
                response.headers[k] = v
                
            # Record metrics
            request_duration = time.time() - start_time
            REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status_code=status_code).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(request_duration)
            RATE_LIMIT_REMAINING.labels(client_id=client_id).set(rate_limiter.get_remaining(client_id))
                
            return response
        
        model = get_model()
        
        # Validate batch request
        if not batch_request.texts:
            raise ValueError("No texts provided for processing")
            
        # Process the batch of texts
        batch_size = str(len(batch_request.texts))
        
        # Auto-detect language or use specified
        text_language = batch_request.language.value
        if text_language == "auto":
            # Use the most common detected language in the batch
            lang_counts = {}
            for text in batch_request.texts:
                detected = detect_language(text)
                LANGUAGE_DETECTION_COUNT.labels(detected_language=detected).inc()
                lang_counts[detected] = lang_counts.get(detected, 0) + 1
            text_language = max(lang_counts.items(), key=lambda x: x[1])[0]
        
        # Measure model prediction time with Prometheus
        with MODEL_PREDICTION_DURATION.labels(batch_size=batch_size).time():
            results = []
            for text in batch_request.texts:
                try:
                    # Preprocess the text
                    processed_text = preprocess_text(
    """
    # Check rate limit
    allowed, rate_limit_headers = check_rate_limit(req)
    if not allowed:
        logger.warning(f"Rate limit exceeded for client: {get_client_id(req)}")
        response = JSONResponse(
            status_code=429,
            content=ErrorResponse(
                detail="Rate limit exceeded. Try again later.",
                error_type="RateLimitExceeded",
                timestamp=datetime.datetime.now().isoformat()
            ).dict()
        )
        # Add rate limit headers
        for k, v in rate_limit_headers.items():
            response.headers[k] = v
        return response
        
    start_time = time.time()
    detected_language = None
    
    try:
        # Validate input text
        is_valid, error_message = validate_text(request.text)
        if not is_valid:
            raise ValueError(f"Invalid input text: {error_message}")
        
        # Auto-detect language if needed
        lang = request.language.value  # Convert from enum to string
        if lang == "auto":
            detected_language = detect_language(request.text)
            lang = detected_language
            logger.debug(f"Detected language: {lang}")
        
        # Preprocess text
        logger.debug(f"Preprocessing text: {request.text[:50]}...")
        processed_text = preprocess_text(request.text, lang)
        
        # Get prediction
        logger.debug("Predicting emotions")
        result = model.predict(processed_text)
        
        # Add additional information
        processing_time_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time_ms
        
        if detected_language:
            result["language_detected"] = detected_language
            
        logger.info(f"Processed text in {processing_time_ms:.2f}ms, dominant emotion: {result['dominant_emotion']}")
        
        # Prepare response with rate limit headers
        response = JSONResponse(
            status_code=200,
            content=result
        )
        
        # Add rate limit headers
        for k, v in rate_limit_headers.items():
            response.headers[k] = v
            
        return response
        
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.post("/batch", response_model=JobStatus, responses={
    202: {"description": "Batch job accepted and processing started"},
    400: {"model": ErrorResponse, "description": "Invalid input"},
    429: {"model": ErrorResponse, "description": "Too many requests"},
    500: {"model": ErrorResponse, "description": "Server error during job creation"}
})
async def create_batch_job(
    request: BatchRequest, 
    req: Request,
    background_tasks: BackgroundTasks,
    model: EmotionDetectionModel = Depends(get_model)
):
    """
    Create a new batch processing job.
    
    Returns a job ID that can be used to check the status and retrieve results.
    """
    # Check rate limit
    allowed, rate_limit_headers = check_rate_limit(req)
    if not allowed:
        logger.warning(f"Rate limit exceeded for client: {get_client_id(req)}")
        response = JSONResponse(
            status_code=429,
            content=ErrorResponse(
                detail="Rate limit exceeded. Try again later.",
                error_type="RateLimitExceeded",
                timestamp=datetime.datetime.now().isoformat()
            ).dict()
        )
        # Add rate limit headers
        for k, v in rate_limit_headers.items():
            response.headers[k] = v
        return response
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        created_time = time.time()
        
        # Validate batch request
        if not request.texts:
            raise ValueError("No texts provided for processing")
        
        # Get batch size, use default if not provided
        batch_size = request.batch_size if request.batch_size is not None else 16
        
        # Create initial job record
        jobs[job_id] = {
            "status": "pending",
            "total_texts": len(request.texts),
            "completed_texts": 0,
            "results": [],
            "created_at": created_time,
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "client_id": get_client_id(req),
            "estimated_completion_time": None,
            "error": None,
            "language": request.language.value
        }
        
        # Schedule automatic cleanup after 1 hour
        background_tasks.add_task(cleanup_job, job_id, 3600)
        
        # Start processing
        logger.info(f"Creating batch job {job_id} with {len(request.texts)} texts")
        
        # If client requested to wait for results, process synchronously
        if request.wait_for_results:
            # Only allow this for small batches
            if len(request.texts) > 50:
                raise ValueError("wait_for_results is only supported for batches with up to 50 texts")
                
            # Process job directly
            await process_batch_job(
                job_id=job_id,
                texts=request.texts,
                language=request.language.value,
                batch_size=batch_size,
                model=model
            )
            
            # Return job status with results
            job = jobs[job_id]
            return BatchJobResponse(
                job_id=job_id,
                status=job["status"],
                total_texts=job["total_texts"],
                completed_texts=job["completed_texts"],
                progress=100.0,
                results=job["results"] if job["status"] == "completed" else None,
                error=job.get("error"),
                processing_time=job["completed_at"] - job["started_at"] if job["completed_at"] else None
            )
        else:
            # Process job in background
            background_tasks.add_task(
                process_batch_job,
                job_id=job_id,
                texts=request.texts,
                language=request.language.value,
                batch_size=batch_size,
                model=model
            )
            
            # Calculate estimated completion time
            estimated_seconds_per_text = 0.1  # Initial estimate
            estimated_seconds = estimated_seconds_per_text * len(request.texts)
            estimated_completion_time = datetime.datetime.fromtimestamp(time.time() + estimated_seconds).isoformat()
            
            # Update job with estimated completion time
            jobs[job_id]["estimated_completion_time"] = estimated_completion_time
            
            # Return job status
            response = JSONResponse(
                status_code=202,  # Accepted
                content=JobStatus(
                    job_id=job_id,
                    status="pending",
                    total_texts=len(request.texts),
                    completed_texts=0,
                    progress=0.0,
                    created_at=datetime.datetime.fromtimestamp(created_time).isoformat(),
                    estimated_completion_time=estimated_completion_time
                ).dict()
            )
            
            # Add rate limit headers
            for k, v in rate_limit_headers.items():
                response.headers[k] = v
            
            # Add Location header for the status endpoint
            response.headers["Location"] = f"/api/v1/batch/{job_id}"
            
            return response
        
    except ValueError as e:
        logger.error(f"Invalid batch request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating batch job: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating batch job: {str(e)}")


@router.get("/batch/{job_id}", response_model=BatchJobResponse, responses={
    200: {"description": "Job status retrieved successfully"},
    404: {"model": ErrorResponse, "description": "Job not found"},
    500: {"model": ErrorResponse, "description": "Server error retrieving job status"}
})
async def get_batch_job(job_id: str, req: Request):
    """
    Get the status and results of a batch processing job.
    
    Returns the job status and, if completed, the processing results.
    """
    try:
        # Check if job exists
        if job_id not in jobs:
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Get job data
        job = jobs[job_id]
        
        # Verify client has access to this job
        client_id = get_client_id(req)
        if job.get("client_id") and job["client_id"] != client_id:
            logger.warning(f"Client {client_id} attempted to access job {job_id} created by {job['client_id']}")
            # Just pretend the job doesn't exist for security
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Calculate progress percentage
        if job["total_texts"] > 0:
            progress = (job["completed_texts"] / job["total_texts"]) * 100
        else:
            progress = 0
            
        # Calculate processing time if completed
        processing_time = None
        if job["completed_at"] and job["started_at"]:
            processing_time = job["completed_at"] - job["started_at"]
            
        # Prepare timestamps for response
        created_at = datetime.datetime.fromtimestamp(job["created_at"]).isoformat() if job["created_at"] else None
        started_at = datetime.datetime.fromtimestamp(job["started_at"]).isoformat() if job["started_at"] else None
        completed_at = datetime.datetime.fromtimestamp(job["completed_at"]).isoformat() if job["completed_at"] else None
            
        # Prepare response
        response = BatchJobResponse(
            job_id=job_id,
            status=job["status"],
            total_texts=job["total_texts"],
            completed_texts=job["completed_texts"],
            progress=progress,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            processing_time=processing_time,
            estimated_completion_time=job.get("estimated_completion_time")
        )
        
        # Add results if job is completed
        if job["status"] == "completed":
            response.results = job["results"]
        
        # Add error if job failed
        if job["status"] == "failed" and "error" in job:
            response.error = job["error"]
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")


@router.delete("/batch/{job_id}", response_model=Dict[str, Any], responses={
    200: {"description": "Job deleted successfully"},
    404: {"model": ErrorResponse, "description": "Job not found"},
    500: {"model": ErrorResponse, "description": "Server error during job deletion"}
})
async def delete_batch_job(job_id: str, req: Request):
    """
    Delete a batch processing job and its results.
    
    This operation cannot be undone.
    """
    try:
        # Check if job exists
        if job_id not in jobs:
            logger.warning(f"Job not found for deletion: {job_id}")
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Get job data
        job = jobs[job_id]
        
        # Verify client has access to this job
        client_id = get_client_id(req)
        if job.get("client_id") and job["client_id"] != client_id:
            logger.warning(f"Client {client_id} attempted to delete job {job_id} created by {job['client_id']}")
            # Just pretend the job doesn't exist for security
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
            
        # Delete the job
        status = job["status"]
        del jobs[job_id]
        
        logger.info(f"Deleted job {job_id} with status {status}")
        
        return {
            "job_id": job_id,
            "status": "deleted",
            "message": "Job and results have been deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")


async def process_batch_job(job_id: str, texts: List[str], language: str, batch_size: int, model: EmotionDetectionModel):
    """
    Process a batch job in the background.
    
    Updates the job status and results as processing progresses.
    """
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = time.time()
        
        logger.info(f"Starting batch job {job_id} with {len(texts)} texts")
        
        # Process in batches with progress tracking
        try:
            # Set processing parameters
            use_batch_size = batch_size if batch_size is not None else 16
            
            # Initialize time tracking variables
            start_time = time.time()
            time_per_text = []
            
            # Process the texts in smaller batches to update progress
            results = []
            for i in range(0, len(texts), use_batch_size):
                # Check if job has been deleted
                if job_id not in jobs:
                    logger.warning(f"Job {job_id} was deleted during processing. Aborting.")
                    return
                
                # Get the current batch
                batch_end = min(i + use_batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                batch_size_actual = len(batch_texts)
                
                # Process the batch
                batch_start_time = time.time()
                
                # Auto-detect language or use specified
                text_language = language
                if language == "auto":
                    # Use the most common detected language in the batch
                    lang_counts = {}
                    for text in batch_texts:
                        detected = detect_language(text)
                        lang_counts[detected] = lang_counts.get(detected, 0) + 1
                    text_language = max(lang_counts.items(), key=lambda x: x[1])[0]
                
                # Preprocess the batch
                processed_texts = [preprocess_text(text, text_language) for text in batch_texts]
                
                # Get predictions
                batch_results = []
                for j, processed_text in enumerate(processed_texts):
                    try:
                        result = model.predict(processed_text)
                        result["text"] = batch_texts[j]  # Use original text
                        
                        # Add language detected
                        if language == "auto":
                            result["language_detected"] = detect_language(batch_texts[j])
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing text {i+j} in job {job_id}: {str(e)}")
                        # Add error result to maintain ordering
                        error_result = {
                            "text": batch_texts[j],
                            "emotions": {},
                            "dominant_emotion": "Unknown",
                            "error": str(e),
                            "model_version": model.metadata.get("version", "unknown")
                        }
                        batch_results.append(error_result)
                
                # Add batch results to overall results
                results.extend(batch_results)
                
                # Calculate time statistics
                batch_time = time.time() - batch_start_time
                avg_time_per_text = batch_time / batch_size_actual
                time_per_text.append(avg_time_per_text)
                
                # Update progress
                completed_so_far = len(results)
                remaining_texts = len(texts) - completed_so_far
                progress_pct = (completed_so_far / len(texts)) * 100
                
                # Calculate moving average of time per text for better estimation
                avg_time_per_text_overall = sum(time_per_text) / len(time_per_text)
                
                # Calculate remaining time
                remaining_time = remaining_texts * avg_time_per_text_overall
                
                # Calculate estimated completion time
                estimated_completion_time = datetime.datetime.fromtimestamp(
                    time.time() + remaining_time
                ).isoformat()
                
                # Check if job still exists (might have been deleted)
                if job_id not in jobs:
                    logger.warning(f"Job {job_id} was deleted during processing. Aborting.")
                    return
                
                # Update job progress
                jobs[job_id]["completed_texts"] = completed_so_far
                jobs[job_id]["progress"] = progress_pct
                jobs[job_id]["estimated_completion_time"] = estimated_completion_time
                
                # Log progress
                elapsed = time.time() - start_time
                texts_per_second = completed_so_far / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Job {job_id}: {completed_so_far}/{len(texts)} texts processed "
                    f"({progress_pct:.1f}%), {texts_per_second:.1f} texts/sec, "
                    f"est. completion: {estimated_completion_time}"
                )
                
                # Add a small delay to avoid overloading the system and allow other requests to process
                await asyncio.sleep(0.01)
            
            # Update job with results
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_texts"] = len(texts)
            jobs[job_id]["results"] = results
            jobs[job_id]["completed_at"] = time.time()
            jobs[job_id]["progress"] = 100.0
            
            # Calculate total processing time
            processing_time = jobs[job_id]["completed_at"] - jobs[job_id]["started_at"]
            
            logger.info(
                f"Completed batch job {job_id} in {processing_time:.2f}s, "
                f"average: {(processing_time / len(texts)):.3f}s per text"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch job {job_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Update job as failed
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = time.time()
            
            # Save partial results if any
            if len(results) > 0:
                jobs[job_id]["results"] = results
                jobs[job_id]["completed_texts"] = len(results)
                jobs[job_id]["progress"] = (len(results) / len(texts)) * 100
                
                logger.info(f"Saved {len(results)}/{len(texts)} partial results for failed job {job_id}")
            
    except Exception as e:
        logger.error(f"Unexpected error in batch processing for job {job_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Unexpected error: {str(e)}"
            jobs[job_id]["completed_at"] = time.time()


async def cleanup_job(job_id: str, ttl_seconds: int = 3600):
    """
    Automatically clean up a job after the specified time to live.
    
    Args:
        job_id: The ID of the job to clean up
        ttl_seconds: Time to live in seconds (default: 1 hour)
    """
    try:
        # Wait for the TTL period
        await asyncio.sleep(ttl_seconds)
        
        # Check if the job still exists
        if job_id in jobs:
            job = jobs[job_id]
            
            # Check if job is still active
            if job["status"] in ["pending", "processing"]:
                logger.warning(f"Job {job_id} is still active after TTL period. Marking as failed.")
                job["status"] = "failed"
                job["error"] = "Job timed out"
                job["completed_at"] = time.time()
            
            # Log cleanup
            status = job["status"]
            logger.info(f"Cleaning up job {job_id} with status {status} after TTL period of {ttl_seconds}s")
            
            # Delete the job
            del jobs[job_id]
    except Exception as e:
        logger.error(f"Error cleaning up job {job_id}: {str(e)}")


async def update_job_progress(job_id: str, completed: int, total: int, error: Optional[str] = None):
    """
    Update the progress of a job.
    
    Args:
        job_id: The ID of the job to update
        completed: Number of completed texts
        total: Total number of texts
        error: Optional error message
    """
    if job_id not in jobs:
        logger.warning(f"Attempted to update progress for non-existent job {job_id}")
        return
        
    # Calculate progress percentage
    progress = (completed / total) * 100 if total > 0 else 0
    
    # Update job
    jobs[job_id]["completed_texts"] = completed
    jobs[job_id]["progress"] = progress
    
    # Update status if all texts are processed
    if completed >= total and jobs[job_id]["status"] == "processing":
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = time.time()
    
    # Update error if provided
    if error:
        jobs[job_id]["error"] = error
        if error and jobs[job_id]["status"] != "failed":
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["completed_at"] = time.time()


async def set_job_status(job_id: str, status: str, error: Optional[str] = None):
    """
    Update the status of a job.
    
    Args:
        job_id: The ID of the job to update
        status: New status (pending, processing, completed, failed)
        error: Optional error message
    """
    if job_id not in jobs:
        logger.warning(f"Attempted to update status for non-existent job {job_id}")
        return
        
    # Update job status
    jobs[job_id]["status"] = status
    
    # Update timestamps based on status
    current_time = time.time()
    if status == "processing" and not jobs[job_id]["started_at"]:
        jobs[job_id]["started_at"] = current_time
    elif status in ["completed", "failed"] and not jobs[job_id]["completed_at"]:
        jobs[job_id]["completed_at"] = current_time
    
    # Update error if provided
    if error:
        jobs[job_id]["error"] = error


async def cache_job_results(job_id: str, results: List[Dict[str, Any]]):
    """
    Cache the results for a job.
    
    Args:
        job_id: The ID of the job
        results: List of result dictionaries
    """
    if job_id not in jobs:
        logger.warning(f"Attempted to cache results for non-existent job {job_id}")
        return
        
    # Cache results
    jobs[job_id]["results"] = results
    
    # Update completed count
    jobs[job_id]["completed_texts"] = len(results)
    
    # Update progress
    if jobs[job_id]["total_texts"] > 0:
        jobs[job_id]["progress"] = (len(results) / jobs[job_id]["total_texts"]) * 100


# Schedule periodic cleanup of old jobs
@router.on_startup
async def start_jobs_cleanup():
    """Start a background task to periodically clean up old jobs."""
    asyncio.create_task(periodic_jobs_cleanup())


async def periodic_jobs_cleanup():
    """Periodically clean up old jobs that have been completed or failed."""
    cleanup_interval = 3600  # 1 hour
    job_ttl = 24 * 3600  # 24 hours
    
    while True:
        try:
            # Wait for the cleanup interval
            await asyncio.sleep(cleanup_interval)
            
            # Get current time
            current_time = time.time()
            
            # Find jobs to clean up
            jobs_to_clean = []
            for job_id, job in jobs.items():
                # Check if job is old enough to clean up
                if job["status"] in ["completed", "failed"]:
                    if job["completed_at"] and (current_time - job["completed_at"]) > job_ttl:
                        jobs_to_clean.append(job_id)
                elif job["status"] == "pending":
                    if job["created_at"] and (current_time - job["created_at"]) > job_ttl:
                        jobs_to_clean.append(job_id)
                elif job["status"] == "processing":
                    if job["started_at"] and (current_time - job["started_at"]) > (2 * job_ttl):
                        # Processing jobs get a longer grace period before cleanup
                        jobs_to_clean.append(job_id)
            
            # Clean up old jobs
            for job_id in jobs_to_clean:
                if job_id in jobs:
                    logger.info(f"Cleaning up old job {job_id} with status {jobs[job_id]['status']}")
                    del jobs[job_id]
            
            if jobs_to_clean:
                logger.info(f"Cleaned up {len(jobs_to_clean)} old jobs. {len(jobs)} jobs remaining.")
            
        except Exception as e:
            logger.error(f"Error in periodic jobs cleanup: {str(e)}")
            # Continue loop even if there's an error
