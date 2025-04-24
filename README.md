# Multilingual Emotion Detection

A Python-based system for detecting emotions in multilingual text, supporting English and Hindi languages.

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
   venv\Scriptsctivate    # Windows
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
```

### Batch Processing
```python
# Analyze multiple texts
texts = [
    'I am feeling very happy today!',
    'मुझे आज बहुत खुशी है।'
]
results = model.predict_batch(texts)
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

## Project Dependencies

- transformers
- torch
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- pytest (for testing)

## License

MIT License
