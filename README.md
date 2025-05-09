# Multilingual Emotion Detection

A Python-based system for detecting emotions in multilingual text, supporting English and Hindi languages.

## Recent Improvements

The system has been significantly enhanced with the following improvements:

1. **Model Improvements**
   - Fixed save/load functionality with proper error handling
   - Added model versioning and metadata tracking
   - Implemented checkpoint mechanism for model states
   - Optimized memory usage for large batch processing
   - Added auto-adjusting batch sizes based on available memory

2. **Multilingual Processing Enhancements**
   - Improved language detection accuracy
   - Enhanced Hindi text normalization and preprocessing
   - Added support for mixed script text (Hinglish)
   - Implemented robust transliteration between Latin and Devanagari scripts
   - Added handling for edge cases in both languages

3. **Robustness Improvements**
   - Comprehensive error handling throughout the codebase
   - Structured logging for easier debugging
   - Progress tracking for long-running operations
   - Proper cleanup of resources after processing
   - Validation for all inputs with useful error messages

4. **Testing**
   - Comprehensive unit tests for all components
   - Integration tests for the complete pipeline
   - Test fixtures and mock data for consistent testing
   - Edge case handling tests
   - Performance benchmarking tests

## Features

- Multilingual support (English and Hindi)
- Emotion detection with 6 basic emotions: Joy, Sadness, Anger, Fear, Surprise, and Neutral
- Pre-trained transformer models (XLM-RoBERTa)
- Comprehensive evaluation metrics and visualization
- Batch processing capability

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── main.py           # Main entry point
│   ├── preprocessor.py   # Text preprocessing
│   ├── model.py         # Emotion detection model
│   └── evaluation.py    # Evaluation metrics
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

## API Documentation

The Multilingual Emotion Detection system provides a RESTful API built with FastAPI. The API allows you to detect emotions in text through both single-text and batch processing endpoints.

### API Overview

- Base URL: `http://localhost:8000` (default when running locally)
- API Version: v1
- API Prefix: `/api/v1`
- Content Type: JSON

### API Endpoints

#### 1. Single Text Analysis

**Endpoint**: `POST /api/v1/detect`

Analyzes a single text and returns emotion detection results.

**Request Format**:
```json
{
  "text": "I am feeling very happy today!",
  "language": "auto"  // Optional: "auto", "en", "hi"
}
```

**Response Format**:
```json
{
  "emotions": {
    "joy": 0.86,
    "sadness": 0.03,
    "anger": 0.01,
    "fear": 0.02,
    "surprise": 0.05,
    "neutral": 0.03
  },
  "dominant_emotion": "joy",
  "processing_time_ms": 124.5,
  "model_version": "1.0.0",
  "language_detected": "en"  // Only included when language="auto"
}
```

**Status Codes**:
- 200: Successful analysis
- 400: Invalid input
- 429: Rate limit exceeded
- 500: Server error

#### 2. Batch Processing

**Endpoint**: `POST /api/v1/batch`

Creates a batch processing job for multiple texts.

**Request Format**:
```json
{
  "texts": [
    "I am feeling very happy today!",
    "मुझे आज बहुत खुशी है।",
    "This makes me so angry!"
  ],
  "language": "auto",      // Optional: "auto", "en", "hi"
  "batch_size": 16,        // Optional: Processing batch size
  "wait_for_results": false  // Optional: Wait for results (only for small batches)
}
```

**Response Format** (for `wait_for_results=false`):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "total_texts": 3,
  "completed_texts": 0,
  "progress": 0.0,
  "created_at": "2023-04-01T10:30:00",
  "estimated_completion_time": "2023-04-01T10:30:05"
}
```

**Status Codes**:
- 202: Batch job accepted and started
- 400: Invalid input
- 429: Rate limit exceeded
- 500: Server error

#### 3. Batch Job Status

**Endpoint**: `GET /api/v1/batch/{job_id}`

Retrieves the status of a batch processing job.

**Response Format** (job in progress):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "total_texts": 3,
  "completed_texts": 1,
  "progress": 33.33,
  "created_at": "2023-04-01T10:30:00",
  "started_at": "2023-04-01T10:30:01",
  "estimated_completion_time": "2023-04-01T10:30:05"
}
```

**Response Format** (job completed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_texts": 3,
  "completed_texts": 3,
  "progress": 100.0,
  "created_at": "2023-04-01T10:30:00",
  "started_at": "2023-04-01T10:30:01",
  "completed_at": "2023-04-01T10:30:03",
  "processing_time": 2.0,
  "results": [
    {
      "text": "I am feeling very happy today!",
      "emotions": {
        "joy": 0.86,
        "sadness": 0.03,
        "anger": 0.01,
        "fear": 0.02,
        "surprise": 0.05,
        "neutral": 0.03
      },
      "dominant_emotion": "joy",
      "model_version": "1.0.0",
      "language_detected": "en"
    },
    // Additional results...
  ]
}
```

**Status Codes**:
- 200: Request successful
- 404: Job not found
- 500: Server error

#### 4. Delete Batch Job

**Endpoint**: `DELETE /api/v1/batch/{job_id}`

Deletes a batch processing job and its results.

**Response Format**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "deleted",
  "message": "Job and results have been deleted"
}
```

**Status Codes**:
- 200: Job deleted successfully
- 404: Job not found
- 500: Server error

#### 5. Health Check

**Endpoint**: `GET /health`

Provides system health information.

**Response Format**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "request_count": 5000,
  "success_rate": 0.998,
  "model_loaded": true
}
```

#### 6. Statistics

**Endpoint**: `GET /stats`

Provides detailed API usage statistics.

**Response Format**:
```json
{
  "uptime_seconds": 3600,
  "requests": {
    "total": 5000,
    "successful": 4990,
    "failed": 10,
    "success_rate": 0.998
  },
  "performance": {
    "average_response_time": 0.125,
    "total_processing_time": 625.0
  },
  "endpoints": {
    "POST /api/v1/detect": 4500,
    "POST /api/v1/batch": 500
  },
  "errors": {
    "ValueError": 5,
    "RateLimitExceeded": 3,
    "HTTPException": 2
  }
}
```

### Rate Limiting

The API implements rate limiting to ensure fair usage:

- Default limit: 100 requests per minute per client

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1680346800
```

When a rate limit is exceeded, the API returns HTTP 429 Too Many Requests.

### Error Handling

All error responses follow a consistent format:

```json
{
  "detail": "Error message explaining what went wrong",
  "error_type": "ValueType",
  "timestamp": "2023-04-01T10:30:00"
}
```

Each response also includes a unique request ID in the `X-Request-ID` header which can be used for troubleshooting.

### Example Usage

#### cURL

**Single Text Analysis:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/detect' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I am feeling very happy today!",
  "language": "auto"
}'
```

**Batch Processing:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/batch' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["I am happy", "I am sad", "I am angry"],
  "language": "auto"
}'
```

**Check Batch Status:**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/batch/550e8400-e29b-41d4-a716-446655440000'
```

#### Python

```python
import requests
import json

# Single text analysis
response = requests.post(
    "http://localhost:8000/api/v1/detect",
    json={"text": "I am feeling very happy today!", "language": "auto"}
)
print(json.dumps(response.json(), indent=2))

# Batch processing
batch_response = requests.post(
    "http://localhost:8000/api/v1/batch",
    json={
        "texts": ["I am happy", "I am sad", "I am angry"],
        "language": "auto"
    }
)
job_id = batch_response.json()["job_id"]

# Check batch status
status_response = requests.get(f"http://localhost:8000/api/v1/batch/{job_id}")
print(json.dumps(status_response.json(), indent=2))
```

### Running the API Server

To run the API server locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.main:app --reload
```

Once started, the API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multilingual_emotion_detection
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/macOS
   venv\Scripts\activate    # Windows
   ```
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Single Text Analysis
```python
from src.model import EmotionDetectionModel

# Initialize model
model = EmotionDetectionModel()

# Analyze text
text = 'I am feeling very happy today!'
result = model.predict(text)
print(result)

# With language specified explicitly
hindi_text = 'मुझे आज बहुत खुशी है।'
result_hi = model.predict(hindi_text)
print(result_hi)

# With Hinglish text (Hindi written in Latin script)
hinglish_text = 'Main aaj bahut khush hoon.'
result_hinglish = model.predict(hinglish_text)  # Will be automatically detected and transliterated
print(result_hinglish)
```
```

### Batch Processing
```python
# Analyze multiple texts
texts = [
    'I am feeling very happy today!',
    'मुझे आज बहुत खुशी है।',
    'Main bahut udaas hoon.'  # Hinglish
]

# Simple batch processing
results = model.predict_batch(texts)
print(results)

# With memory-efficient batch processing and progress tracking
results = model.predict_batch(
    texts,
    batch_size=16,              # Set batch size
    show_progress=True,         # Show progress bar
    auto_adjust_batch=True      # Automatically adjust batch size based on memory
)
print(results)
```

### Command Line Interface
```bash
python -m src.main --text 'I am feeling happy today!'
python -m src.main --dataset path/to/dataset.csv --output results.csv
```

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_preprocessor.py
pytest tests/test_model.py
pytest tests/test_evaluation.py
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

## Memory Management

For large datasets, the system now includes automatic memory management:

```python
# Dynamically adjust batch size based on available memory
optimal_batch_size = model.estimate_optimal_batch_size()
print(f"Optimal batch size: {optimal_batch_size}")

# Monitor memory usage
memory_info = model.get_memory_usage()
print(f"System memory usage: {memory_info['system_percent']}%")
if 'cuda_used_gb' in memory_info:
    print(f"GPU memory used: {memory_info['cuda_used_gb']:.2f} GB")
```

## Model Checkpointing

The system now supports model checkpointing for reliable state saving and recovery:

```python
# Save a checkpoint
checkpoint_path = model.create_checkpoint("checkpoints")
print(f"Checkpoint saved to: {checkpoint_path}")

# Load from checkpoint
model.load_checkpoint(checkpoint_path)

# Save model with version
model.save_model("models/my_model", version="1.2.0")
```

## Handling Hinglish (Mixed Script Text)

The system now handles mixed script text (Hinglish) more effectively:

```python
from src.preprocessor import process_hinglish_text, transliterate_to_devanagari

# Process Hinglish text (Hindi written in Latin script)
hinglish = "Main office mein kaam karta hoon"
processed = process_hinglish_text(hinglish)
print(processed)  # Will contain Devanagari characters

# Transliterate specific text
transliterated = transliterate_to_devanagari("namaste")
print(transliterated)  # नमस्ते
```

## Project Dependencies

- transformers
- torch
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- tqdm (for progress tracking)
- psutil (for memory monitoring)
- langdetect (for language detection)
- pytest (for testing)


## Testing

Run comprehensive tests for all modules:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src tests/
```


## License
- numpy
- matplotlib
- seaborn
- pytest (for testing)

## Environment Variables

The API supports the following environment variables for configuration:

| Variable | Description | Default |
-----------|-------------|---------|
`MODEL_PATH` | Path to the emotion detection model | `models/emotion_model` |
`LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
`API_HOST` | Host to bind the API server | `0.0.0.0` |
`API_PORT` | Port to bind the API server | `8000` |
`RATE_LIMIT` | Number of requests allowed per minute | `100` |
`RATE_LIMIT_WINDOW` | Rate limiting window in seconds | `60` |
`BATCH_SIZE` | Default batch size for processing | `16` |
`MAX_BATCH_SIZE` | Maximum allowed batch size | `100` |
`JOB_TTL` | Time to live for batch jobs in seconds | `86400` (24 hours) |
`CORS_ORIGINS` | Comma-separated list of allowed origins for CORS | `*` |

These can be set in a `.env` file in the project root or as system environment variables.

## License

MIT License
