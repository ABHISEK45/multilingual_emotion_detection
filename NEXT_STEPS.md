# Next Steps for Multilingual Emotion Detection

This document outlines detailed plans for the highest-priority improvements to build on the now-stable codebase. These improvements are organized in order of priority and impact.

## 1. API Implementation with FastAPI

### Why FastAPI?
- Async support out of the box
- Automatic OpenAPI documentation
- Type validation with Pydantic
- High performance

### Implementation Plan:

1. **Setup**:
   ```bash
   pip install fastapi uvicorn
   ```

2. **Create API Module Structure**:
   ```
   src/
   ├── api/
   │   ├── __init__.py
   │   ├── main.py        # FastAPI application
   │   ├── models.py      # Pydantic models for requests/responses
   │   ├── endpoints.py   # API endpoint definitions
   │   └── utils.py       # Helper functions
   ```

3. **Core API Implementation (`src/api/main.py`)**:
   ```python
   from fastapi import FastAPI, HTTPException, BackgroundTasks
   from .models import TextRequest, BatchRequest, EmotionResponse, BatchJobResponse
   from .endpoints import router
   import logging

   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   # Create FastAPI app
   app = FastAPI(
       title="Multilingual Emotion Detection API",
       description="API for detecting emotions in multilingual text",
       version="1.0.0"
   )

   # Include routers
   app.include_router(router, prefix="/api/v1")

   # Root endpoint
   @app.get("/")
   async def root():
       return {"message": "Welcome to the Multilingual Emotion Detection API"}
   ```

4. **Define API Models (`src/api/models.py`)**:
   ```python
   from pydantic import BaseModel, Field, validator
   from typing import List, Dict, Optional, Any

   class TextRequest(BaseModel):
       text: str = Field(..., min_length=1, description="Text to analyze")
       language: Optional[str] = Field("auto", description="Language code (en, hi, auto)")

       @validator('language')
       def validate_language(cls, v):
           if v not in ["en", "hi", "auto"]:
               raise ValueError("Language must be one of: en, hi, auto")
           return v

   class BatchRequest(BaseModel):
       texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
       language: Optional[str] = Field("auto", description="Language code (en, hi, auto)")
       batch_size: Optional[int] = Field(16, gt=0, description="Batch size for processing")

   class EmotionResponse(BaseModel):
       text: str
       emotions: Dict[str, float]
       dominant_emotion: str
       model_version: str

   class BatchJobResponse(BaseModel):
       job_id: str
       status: str
       total_texts: int
       completed_texts: int = 0
       results: Optional[List[EmotionResponse]] = None
   ```

5. **Define Endpoints (`src/api/endpoints.py`)**:
   ```python
   from fastapi import APIRouter, HTTPException, BackgroundTasks
   from .models import TextRequest, BatchRequest, EmotionResponse, BatchJobResponse
   from src.model import EmotionDetectionModel
   from src.preprocessor import preprocess_text, detect_language
   from typing import Dict, List
   import uuid
   import logging
   import asyncio

   router = APIRouter()
   
   # Store for background jobs
   jobs = {}

   # Initialize model
   model = EmotionDetectionModel()

   @router.post("/detect", response_model=EmotionResponse)
   async def detect_emotion(request: TextRequest):
       try:
           # Auto-detect language if needed
           lang = request.language
           if lang == "auto":
               lang = detect_language(request.text)
               
           # Preprocess text
           processed_text = preprocess_text(request.text, lang)
           
           # Get prediction
           result = model.predict(processed_text)
           return result
       except Exception as e:
           logging.error(f"Error processing text: {str(e)}")
           raise HTTPException(status_code=500, detail=str(e))

   @router.post("/batch", response_model=BatchJobResponse)
   async def create_batch_job(request: BatchRequest, background_tasks: BackgroundTasks):
       # Create job ID
       job_id = str(uuid.uuid4())
       
       # Initialize job
       jobs[job_id] = {
           "status": "pending",
           "total_texts": len(request.texts),
           "completed_texts": 0,
           "results": []
       }
       
       # Start background processing
       background_tasks.add_task(process_batch, job_id, request)
       
       return BatchJobResponse(
           job_id=job_id,
           status="pending",
           total_texts=len(request.texts)
       )

   @router.get("/batch/{job_id}", response_model=BatchJobResponse)
   async def get_batch_job(job_id: str):
       if job_id not in jobs:
           raise HTTPException(status_code=404, detail="Job not found")
       
       job = jobs[job_id]
       return BatchJobResponse(
           job_id=job_id,
           status=job["status"],
           total_texts=job["total_texts"],
           completed_texts=job["completed_texts"],
           results=job["results"] if job["status"] == "completed" else None
       )

   async def process_batch(job_id: str, request: BatchRequest):
       try:
           jobs[job_id]["status"] = "processing"
           
           # Process in batches
           results = model.predict_batch(
               request.texts,
               batch_size=request.batch_size,
               show_progress=False
           )
           
           # Update job
           jobs[job_id]["status"] = "completed"
           jobs[job_id]["completed_texts"] = len(request.texts)
           jobs[job_id]["results"] = results
           
       except Exception as e:
           logging.error(f"Error processing batch: {str(e)}")
           jobs[job_id]["status"] = "failed"
           jobs[job_id]["error"] = str(e)
   ```

6. **Run the API**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

7. **Access Documentation**:
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

## 2. Additional Language Support (Bengali and Tamil)

### Implementation Plan:

1. **Update Supported Languages**:
   ```python
   # In src/preprocessor.py
   SUPPORTED_LANGUAGES = {'en', 'hi', 'bn', 'ta'}  # Added Bengali (bn) and Tamil (ta)
   ```

2. **Add Script Detection for New Languages**:
   ```python
   # In src/preprocessor.py
   def detect_language(text: str) -> str:
       # Existing code...
       
       # Add detection for Bengali (range: \u0980-\u09FF)
       bengali_pattern = re.compile(r'[\u0980-\u09FF]')
       bengali_chars = len(re.findall(bengali_pattern, text))
       
       # Add detection for Tamil (range: \u0B80-\u0BFF)
       tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
       tamil_chars = len(re.findall(tamil_pattern, text))
       
       # Check character counts (simplified)
       char_counts = {
           'hi': devanagari_chars,
           'bn': bengali_chars,
           'ta': tamil_chars
       }
       
       # Get the script with most characters
       max_script = max(char_counts.items(), key=lambda x: x[1])
       if max_script[1] > 0 and max_script[1] / len(text.replace(" ", "")) > 0.25:
           return max_script[0]
           
       # Default to English
       return 'en'
   ```

3. **Add Script-Specific Normalization**:
   ```python
   # In src/preprocessor.py
   def normalize_bengali_text(text: str) -> str:
       """Normalize Bengali text for better processing."""
       # Apply Unicode normalization
       text = unicodedata.normalize('NFC', unicodedata.normalize('NFD', text))
       
       # Bengali-specific character normalization
       char_maps = {
           # Add Bengali-specific mappings here
       }
       
       # Apply character mappings
       for char, replacement in char_maps.items():
           text = text.replace(char, replacement)
       
       # Normalize spacing around punctuation
       text = re.sub(r'\s*([।,.!?])\s*', r'\1 ', text)
       text = text.strip()
       
       return text

   def normalize_tamil_text(text: str) -> str:
       """Normalize Tamil text for better processing."""
       # Apply Unicode normalization
       text = unicodedata.normalize('NFC', unicodedata.normalize('NFD', text))
       
       # Tamil-specific character normalization
       char_maps = {
           # Add Tamil-specific mappings here
       }
       
       # Apply character mappings
       for char, replacement in char_maps.items():
           text = text.replace(char, replacement)
       
       # Normalize spacing around punctuation
       text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
       text = text.strip()
       
       return text
   ```

4. **Update Preprocessing Logic**:
   ```python
   # In src/preprocessor.py, modify preprocess_text function
   def preprocess_text(text: str, language: Optional[str] = None) -> str:
       # Existing code...
       
       # Add cases for new languages
       elif language == 'bn':
           # Bengali-specific preprocessing
           text = normalize_bengali_text(text)
           # Keep Bengali Unicode range characters
           text = re.sub(r'[^\u0980-\u09FF\s\.\,\!\?\-]', '', text)
           
       elif language == 'ta':
           # Tamil-specific preprocessing
           text = normalize_tamil_text(text)
           # Keep Tamil Unicode range characters
           text = re.sub(r'[^\u0B80-\u0BFF\s\.\,\!\?\-]', '', text)
   ```

5. **Add Test Cases**:
   ```python
   # In tests/test_preprocessor.py
   def test_detect_bengali():
       """Test detection of Bengali text."""
       bengali_texts = [
           "আমি আজ খুব খুশি।",  # I am very happy today.
           "এটা আমার জীবনের সবচেয়ে খারাপ দিন।"  # This is the worst day of my life.
       ]
       
       for text in bengali_texts:
           assert detect_language(text) == 'bn', f"Failed to detect Bengali: {text}"
           
   def test_detect_tamil():
       """Test detection of Tamil text."""
       tamil_texts = [
           "இன்று நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்.",  # I am very happy today.
           "இது என் வாழ்க்கையின் மிகவும் மோசமான நாள்."  # This is the worst day of my life.
       ]
       
       for text in tamil_texts:
           assert detect_language(text) == 'ta', f"Failed to detect Tamil: {text}"
   ```

## 3. Model Fine-tuning Capabilities

### Implementation Plan:

1. **Create a Fine-tuning Module (`src/train.py`)**:
   ```python
   """
   Model fine-tuning functionality for the emotion detection system.
   """
   
   import os
   import torch
   import numpy as np
   import pandas as pd
   from tqdm import tqdm
   from pathlib import Path
   from typing import Dict, List, Optional, Union, Tuple
   from transformers import (
       AutoModelForSequenceClassification,
       AutoTokenizer,
       Trainer,
       TrainingArguments,
       DataCollatorWithPadding
   )
   from sklearn.model_selection import train_test_split
   from datasets import Dataset
   
   from .model import EMOTION_CLASSES, EmotionDetectionModel
   from .preprocessor import preprocess_text, detect_language
   
   
   class EmotionModelTrainer:
       """Handles fine-tuning of emotion detection models."""
       
       def __init__(self, base_model_path: Optional[str] = None, output_dir: str = "models/fine_tuned"):
           """
           Initialize the trainer.
           
           Args:
               base_model_path: Path to the base model to fine-tune. If None, uses the default model.
               output_dir: Directory to save fine-tuned models.
           """
           self.base_model_path = base_model_path
           self.output_dir = Path(output_dir)
           self.output_dir.mkdir(parents=True, exist_ok=True)
           
           # Initialize base model and tokenizer
           if base_model_path:
               self.model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
               self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
           else:
               # Default model
               self.model_name = "xlm-roberta-base"
               self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
               self.model = AutoModelForSequenceClassification.from_pretrained(
                   self.model_name,
                   num_labels=len(EMOTION_CLASSES)
               )
       
       def prepare_dataset(self, 
                          data_path: str, 
                          text_column: str, 
                          label_column: str,
                          test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
           """
           Prepare dataset for fine-tuning.
           
           Args:
               data_path: Path to dataset file (CSV, JSON)
               text_column: Name of column containing text
               label_column: Name of column containing emotion labels
               test_size: Fraction of data to use for testing
               
           Returns:
               Tuple of (train_dataset, eval_dataset)
           """
           # Load data
           if data_path.endswith('.csv'):
               df = pd.read_csv(data_path)
           elif data_path.endswith('.json'):
               df = pd.read_json(data_path)
           else:
               raise ValueError(f"Unsupported file format: {data_path}")
               
           # Create label mapping
           unique_labels = df[label_column].unique()
           label_map = {label: i for i, label in enumerate(unique_labels)}
           
           # Preprocess data
           df['processed_text'] = df[text_column].apply(
               lambda x: preprocess_text(x, detect_language(x))
           )
           df['label

