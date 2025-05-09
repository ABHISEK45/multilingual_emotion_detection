"""
Pydantic models for the API request and response validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import datetime


class SupportedLanguage(str, Enum):
    """Supported languages for emotion detection."""
    ENGLISH = "en"
    HINDI = "hi"
    AUTO = "auto"


class EmotionName(str, Enum):
    """Emotion categories."""
    JOY = "Joy"
    SADNESS = "Sadness"
    ANGER = "Anger"
    FEAR = "Fear"
    SURPRISE = "Surprise"
    NEUTRAL = "Neutral"


class TextRequest(BaseModel):
    """Request model for single text emotion detection."""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze for emotions (1-5000 characters)"
    )
    language: SupportedLanguage = Field(
        SupportedLanguage.AUTO, 
        description="Language code ('en', 'hi', or 'auto' for detection)"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "I am feeling very happy today!",
                "language": "auto"
            }
        }


class BatchRequest(BaseModel):
    """Request model for batch emotion detection."""
    texts: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=1000,
        description="List of texts to analyze (1-1000 items)"
    )
    language: SupportedLanguage = Field(
        SupportedLanguage.AUTO, 
        description="Language code (en, hi, auto)"
    )
    batch_size: Optional[int] = Field(
        16, 
        gt=0, 
        le=128,
        description="Batch size for processing (1-128)"
    )
    wait_for_results: Optional[bool] = Field(
        False,
        description="Whether to wait for results (if True, request will not return until processing is complete)"
    )

    @validator('texts')
    def validate_texts(cls, v):
        """Validate that texts are non-empty and not too long."""
        if not all(isinstance(text, str) for text in v):
            raise ValueError("All texts must be strings")
        
        if not all(text.strip() for text in v):
            raise ValueError("All texts must be non-empty after trimming whitespace")
            
        if any(len(text) > 5000 for text in v):
            raise ValueError("Texts must not exceed 5000 characters")
            
        return v
        
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I am feeling very happy today!",
                    "This news makes me so sad.",
                    "मुझे आज बहुत खुशी है।",
                    "Main bahut udaas hoon."
                ],
                "language": "auto",
                "batch_size": 16
            }
        }


class EmotionScore(BaseModel):
    """Model for emotion score details."""
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0-1.0)"
    )
    label: EmotionName = Field(
        ..., 
        description="Emotion category"
    )
    description: Optional[str] = Field(
        None, 
        description="Description of the emotion"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "score": 0.85,
                "label": "Joy",
                "description": "Feeling of great pleasure and happiness"
            }
        }


class EmotionResponse(BaseModel):
    """Response model for emotion detection results."""
    text: str = Field(
        ..., 
        description="Original text that was analyzed"
    )
    emotions: Dict[str, float] = Field(
        ..., 
        description="Emotion probabilities (name -> score mapping)"
    )
    scores: Optional[List[EmotionScore]] = Field(
        None, 
        description="Detailed emotion scores with descriptions (sorted by score)"
    )
    dominant_emotion: EmotionName = Field(
        ..., 
        description="Most likely emotion"
    )
    model_version: str = Field(
        ..., 
        description="Version of the model used for prediction"
    )
    language_detected: Optional[str] = Field(
        None, 
        description="Detected language code (if auto-detection was used)"
    )
    processing_time_ms: Optional[float] = Field(
        None, 
        description="Processing time in milliseconds"
    )
    
    @root_validator
    def populate_scores(cls, values):
        """Populate scores list from emotions dictionary if it doesn't exist."""
        emotions = values.get('emotions')
        scores = values.get('scores')
        
        if emotions and not scores:
            scores = [
                EmotionScore(
                    score=score,
                    label=emotion,
                    description=None
                )
                for emotion, score in emotions.items()
            ]
            # Sort by score descending
            scores.sort(key=lambda x: x.score, reverse=True)
            values['scores'] = scores
            
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I am feeling very happy today!",
                "emotions": {
                    "Joy": 0.85,
                    "Sadness": 0.03,
                    "Anger": 0.02,
                    "Fear": 0.02,
                    "Surprise": 0.05,
                    "Neutral": 0.03
                },
                "dominant_emotion": "Joy",
                "model_version": "1.0.0",
                "language_detected": "en",
                "processing_time_ms": 42.5
            }
        }


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    detail: str = Field(
        ..., 
        description="Error description"
    )
    error_type: Optional[str] = Field(
        None, 
        description="Type of error"
    )
    timestamp: Optional[str] = Field(
        None, 
        description="Time when the error occurred"
    )
    request_id: Optional[str] = Field(
        None, 
        description="Request ID for tracking the error"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid input: Text must be a non-empty string",
                "error_type": "ValueError",
                "timestamp": datetime.datetime.now().isoformat(),
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class JobStatus(BaseModel):
    """Model for batch job status information."""
    job_id: str = Field(
        ..., 
        description="Unique job identifier"
    )
    status: str = Field(
        ..., 
        description="Job status (pending, processing, completed, failed)"
    )
    total_texts: int = Field(
        ..., 
        ge=0,
        description="Total number of texts to process"
    )
    completed_texts: int = Field(
        0, 
        ge=0,
        description="Number of texts processed so far"
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if job failed"
    )
    progress: Optional[float] = Field(
        None, 
        ge=0.0,
        le=100.0,
        description="Progress as a percentage (0-100)"
    )
    estimated_completion_time: Optional[str] = Field(
        None, 
        description="Estimated time of completion (ISO 8601 format)"
    )
    created_at: Optional[str] = Field(
        None, 
        description="Time when the job was created (ISO 8601 format)"
    )
    started_at: Optional[str] = Field(
        None, 
        description="Time when the job started processing (ISO 8601 format)"
    )
    completed_at: Optional[str] = Field(
        None, 
        description="Time when the job completed (ISO 8601 format)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "total_texts": 100,
                "completed_texts": 42,
                "progress": 42.0,
                "created_at": datetime.datetime.now().isoformat(),
                "started_at": datetime.datetime.now().isoformat()
            }
        }


class BatchJobResponse(BaseModel):
    """Response model for batch processing jobs."""
    job_id: str = Field(
        ..., 
        description="Unique job identifier"
    )
    status: str = Field(
        ..., 
        description="Job status (pending, processing, completed, failed)"
    )
    total_texts: int = Field(
        ..., 
        ge=0,
        description="Total number of texts to process"
    )
    completed_texts: int = Field(
        0, 
        ge=0,
        description="Number of texts processed so far"
    )
    progress: Optional[float] = Field(
        None, 
        ge=0.0,
        le=100.0,
        description="Progress as a percentage (0-100)"
    )
    results: Optional[List[EmotionResponse]] = Field(
        None, 
        description="Results (only provided when job is completed)"
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if job failed"
    )
    estimated_completion_time: Optional[str] = Field(
        None, 
        description="Estimated time of completion (ISO 8601 format)"
    )
    processing_time: Optional[float] = Field(
        None, 
        ge=0.0,
        description="Total processing time in seconds (only provided when job is completed)"
    )
    
    @validator('results')
    def validate_results_length(cls, v, values):
        """Validate that the number of results matches the total_texts."""
        if v is not None and values.get('total_texts') is not None:
            if len(v) != values['total_texts'] and values['status'] == 'completed':
                raise ValueError(f"Results count ({len(v)}) does not match total_texts ({values['total_texts']})")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "total_texts": 4,
                "completed_texts": 4,
                "progress": 100.0,
                "results": [
                    {
                        "text": "I am feeling very happy today!",
                        "emotions": {
                            "Joy": 0.85,
                            "Sadness": 0.03,
                            "Anger": 0.02,
                            "Fear": 0.02,
                            "Surprise": 0.05,
                            "Neutral": 0.03
                        },
                        "dominant_emotion": "Joy",
                        "model_version": "1.0.0"
                    },

