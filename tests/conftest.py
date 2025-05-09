"""
Test fixtures and mock data for the multilingual emotion detection system.

This module provides shared fixtures and data for use across all test modules.
"""

import os
import json
import pytest
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from src.model import EmotionDetectionModel, EMOTION_CLASSES


@pytest.fixture
def sample_texts():
    """Fixture with sample texts in different languages."""
    return {
        'english': [
            "I am feeling very happy today!",
            "This news makes me so sad.",
            "I'm really angry about what happened.",
            "I'm scared of the dark.",
            "Wow, I'm surprised by the results!",
            "The weather is nice today."
        ],
        'hindi': [
            "मुझे आज बहुत खुशी है।",
            "इस खबर से मुझे बहुत दुःख हुआ।",
            "मुझे इस घटना पर बहुत गुस्सा आ रहा है।",
            "मुझे अंधेरे से डर लगता है।",
            "वाह, मुझे परिणामों से आश्चर्य हुआ!",
            "आज मौसम अच्छा है।"
        ],
        'hinglish': [
            "Main bahut happy feel kar raha hoon.",
            "Yeh news mujhe sad kar deti hai.",
            "Main is incident se angry hoon.",
            "Mujhe dark se dar lagta hai.",
            "Wow, main results se surprised hoon!",
            "Weather aaj acha hai."
        ],
        'invalid': [
            "",
            None,
            123,
            ["This is not a string"],
            {"text": "This is a dictionary"}
        ]
    }


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model directory with required files."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    
    # Create mock model files
    config = {
        "architectures": ["RobertaForSequenceClassification"],
        "model_type": "roberta",
        "num_labels": len(EMOTION_CLASSES)
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    # Create an empty model file
    with open(model_dir / "pytorch_model.bin", "wb") as f:
        f.write(b"mock model data")
    
    # Create metadata
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({
            "version": "1.0.0-test",
            "creation_date": "2025-01-01T00:00:00",
            "last_updated": "2025-01-01T00:00:00"
        }, f)
    
    return str(model_dir)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long)
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.return_value = MagicMock()
    # Mock outputs with logits
    model.return_value.logits = torch.rand((1, len(EMOTION_CLASSES)))
    return model


@pytest.fixture
def mock_emotion_model(monkeypatch, mock_tokenizer, mock_model):
    """Create a mock EmotionDetectionModel instance."""
    monkeypatch.setattr('src.model.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', 
                        lambda *args, **kwargs: mock_model)
    return EmotionDetectionModel()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        'text': [
            "I am feeling very happy today!",
            "This news makes me so sad.",
            "मुझे आज बहुत खुशी है।",
            "इस खबर से मुझे बहुत दुःख हुआ।",
            "Main bahut happy feel kar raha hoon."
        ],
        'language': ['en', 'en', 'hi', 'hi', 'en'],
        'label': ['Joy', 'Sadness', 'Joy', 'Sadness', 'Joy']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset_path(sample_dataset, tmp_path):
    """Save sample dataset to a temporary file and return its path."""
    dataset_path = tmp_path / "sample_dataset.csv"
    sample_dataset.to_csv(dataset_path, index=False)
    return str(dataset_path)


@pytest.fixture
def output_path(tmp_path):
    """Provide a temporary output path."""
    return str(tmp_path / "output.json")


@pytest.fixture
def mock_transform_module():
    """Mock the transformers module."""
    # Create mock module with the necessary components
    transformers_mock = MagicMock()
    transformers_mock.AutoModelForSequenceClassification = MagicMock()
    transformers_mock.AutoTokenizer = MagicMock()
    
    # Configure the mock model
    model_mock = MagicMock()
    model_mock.eval = MagicMock(return_value=None)
    model_mock.to = MagicMock(return_value=model_mock)
    model_mock.save_pretrained = MagicMock()
    
    # Configure mock outputs for prediction
    outputs_mock = MagicMock()
    outputs_mock.logits = torch.tensor([[0.1, 0.5, 0.1, 0.1, 0.1, 0.1]])  # Sadness is highest
    model_mock.return_value = outputs_mock
    
    # Configure the mock tokenizer
    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    tokenizer_mock.save_pretrained = MagicMock()
    
    # Assign to auto classes
    transformers_mock.AutoModelForSequenceClassification.from_pretrained.return_value = model_mock
    transformers_mock.AutoTokenizer.from_pretrained.return_value = tokenizer_mock
    
    return transformers_mock

"""
Common test fixtures for the multilingual emotion detection project.
"""

import os
import pandas as pd
import pytest
import shutil
from pathlib import Path

from src.model import EMOTION_CLASSES


@pytest.fixture
def test_data_dir(tmp_path):
    """
    Create a temporary directory for test outputs.
    
    The directory and its contents are automatically removed after the test is completed.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_texts():
    """
    Sample texts in English and Hindi for testing.
    
    Contains a variety of emotional content in both languages.
    """
    return [
        "I am so happy today!",                            # Joy (English)
        "This is the worst day ever.",                     # Sadness (English)
        "I am feeling very angry right now.",              # Anger (English)
        "I'm afraid of what might happen.",                # Fear (English)
        "Wow! I can't believe this happened!",             # Surprise (English)
        "It's just an ordinary day.",                      # Neutral (English)
        "मुझे आज बहुत खुशी है।",                           # Joy (Hindi) - I am very happy today.
        "यह मेरे जीवन का सबसे बुरा दिन है।",                # Sadness (Hindi) - This is the worst day of my life.
        "मैं इस बात से बहुत गुस्सा हूँ।",                   # Anger (Hindi) - I am very angry about this.
        "मुझे भविष्य के बारे में डर लग रहा है।",             # Fear (Hindi) - I am afraid about the future.
        "वाह! यह तो अविश्वसनीय है!",                        # Surprise (Hindi) - Wow! This is unbelievable!
        "आज एक सामान्य दिन है।"                             # Neutral (Hindi) - Today is a normal day.
    ]


@pytest.fixture
def sample_labels():
    """
    Emotion labels corresponding to the sample_texts.
    
    Labels match the emotions expressed in the sample texts.
    """
    return [
        "Joy",
        "Sadness",
        "Anger",
        "Fear",
        "Surprise",
        "Neutral",
        "Joy",
        "Sadness",
        "Anger",
        "Fear",
        "Surprise",
        "Neutral"
    ]


@pytest.fixture
def sample_dataset(sample_texts, sample_labels):
    """
    Sample dataset as a pandas DataFrame.
    
    Contains sample texts, corresponding emotion labels, and language indicators.
    """
    # Determine language for each text (first 6 are English, next 6 are Hindi)
    languages = ['en'] * 6 + ['hi'] * 6
    
    return pd.DataFrame({
        'text': sample_texts,
        'emotion': sample_labels,
        'language': languages
    })


@pytest.fixture
def prediction_results():
    """
    Sample prediction results in the format returned by the model.predict method.
    
    Each result includes the original text, emotion probabilities, and the dominant emotion.
    """
    return [
        {
            'text': 'I am so happy today!',
            'emotions': {
                'Joy': 0.85,
                'Sadness': 0.03, 
                'Anger': 0.02, 
                'Fear': 0.02, 
                'Surprise': 0.05, 
                'Neutral': 0.03
            },
            'dominant_emotion': 'Joy'
        },
        {
            'text': 'This is the worst day ever.',
            'emotions': {
                'Joy': 0.02,
                'Sadness': 0.80, 
                'Anger': 0.10, 
                'Fear': 0.05, 
                'Surprise': 0.01, 
                'Neutral': 0.02
            },
            'dominant_emotion': 'Sadness'
        },
        {
            'text': 'I am feeling very angry right now.',
            'emotions': {
                'Joy': 0.01,
                'Sadness': 0.10, 
                'Anger': 0.78, 
                'Fear': 0.05, 
                'Surprise': 0.01, 
                'Neutral': 0.05
            },
            'dominant_emotion': 'Anger'
        },
        {
            'text': "I'm afraid of what might happen.",
            'emotions': {
                'Joy': 0.01,
                'Sadness': 0.15, 
                'Anger': 0.04, 
                'Fear': 0.75, 
                'Surprise': 0.02, 
                'Neutral': 0.03
            },
            'dominant_emotion': 'Fear'
        }
    ]


@pytest.fixture
def sample_csv_path(sample_dataset, test_data_dir):
    """
    Create a sample CSV file for testing.
    
    Writes the sample dataset to a CSV file in the test data directory.
    The file is automatically removed after the test is completed.
    """
    csv_path = test_data_dir / "sample_dataset.csv"
    sample_dataset.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_tokenizer():
    """
    Mock tokenizer for testing.
    
    Simulates the behavior of a tokenizer without requiring the actual transformer models.
    """
    class MockTokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
            import torch
            
            # Handle single text or list of texts
            if isinstance(texts, str):
                batch_size = 1
            else:
                batch_size = len(texts)
                
            # Create dummy token IDs and attention mask
            return {
                'input_ids': torch.ones((batch_size, 10), dtype=torch.long),
                'attention_mask': torch.ones((batch_size, 10), dtype=torch.long)
            }
    
    return MockTokenizer()


@pytest.fixture
def mock_model():
    """
    Mock model for testing.
    
    Simulates the behavior of a transformer model without requiring the actual model.
    """
    class MockOutput:
        def __init__(self, batch_size):
            import torch
            import numpy as np
            
            # Create logits that favor different emotions for different inputs
            # for deterministic testing
            logits = torch.zeros((batch_size, len(EMOTION_CLASSES)))
            for i in range(min(batch_size, len(EMOTION_CLASSES))):
                logits[i, i] = 5.0  # High value for one emotion per input
            
            self.logits = logits
    
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            # Do nothing, just simulate setting to eval mode
            pass
            
        def __call__(self, **kwargs):
            # Extract batch size from inputs
            batch_size = kwargs['input_ids'].shape[0]
            return MockOutput(batch_size)
    
    return MockModel()

"""
Common test fixtures for the multilingual emotion detection project.
"""

import os
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.model import EMOTION_CLASSES, EmotionDetectionModel


@pytest.fixture
def sample_texts():
    """Sample texts in English and Hindi for testing."""
    return [
        "I am so happy today!",
        "This is the worst day ever.",
        "I am feeling anxious about the exam.",
        "What a surprise!",
        "मुझे आज बहुत खुशी है।",  # I am very happy today.
        "यह मेरे जीवन का सबसे बुरा दिन है।",  # This is the worst day of my life.
        "मुझे परीक्षा के बारे में चिंता हो रही है।",  # I am feeling worried about the exam.
        "वाह, कितना आश्चर्य!"  # Wow, what a surprise!
    ]


@pytest.fixture
def sample_labels():
    """Sample emotion labels corresponding to sample_texts."""
    return [
        "Joy",
        "Sadness",
        "Fear",
        "Surprise",
        "Joy",
        "Sadness",
        "Fear",
        "Surprise"
    ]


@pytest.fixture
def sample_dataset(sample_texts, sample_labels):
    """Sample dataset as a DataFrame."""
    return pd.DataFrame({
        'text': sample_texts,
        'emotion': sample_labels,
        'language': ['en', 'en', 'en', 'en', 'hi', 'hi', 'hi', 'hi']
    })


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            pass
            
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
            if isinstance(texts, str):
                texts = [texts]
            
            # Create a simple mock encoding
            return {
                'input_ids': torch.ones((len(texts), 10), dtype=torch.long),
                'attention_mask': torch.ones((len(texts), 10), dtype=torch.long)
            }
    
    return MockTokenizer()


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            pass
            
        def __call__(self, **kwargs):
            batch_size = kwargs['input_ids'].shape[0]
            # Return mock logits
            class MockOutput:
                def __init__(self, batch_size):
                    self.logits = torch.randn(batch_size, len(EMOTION_CLASSES))
            
            return MockOutput(batch_size)
    
    return MockModel()


@pytest.fixture
def mock_emotion_model(mock_tokenizer, mock_model):
    """Mock emotion detection model for testing."""
    model = EmotionDetectionModel()
    model.tokenizer = mock_tokenizer
    model.model = mock_model
    return model


@pytest.fixture
def prediction_results():
    """Sample prediction results for testing evaluation functions."""
    return [
        {
            'text': 'I am so happy today!',
            'emotions': {
                'Joy': 0.8, 'Sadness': 0.05, 'Anger': 0.05, 'Fear': 0.03, 'Surprise': 0.05, 'Neutral': 0.02
            },
            'dominant_emotion': 'Joy'
        },
        {
            'text': 'This is the worst day ever.',
            'emotions': {
                'Joy': 0.05, 'Sadness': 0.75, 'Anger': 0.1, 'Fear': 0.05, 'Surprise': 0.03, 'Neutral': 0.02
            },
            'dominant_emotion': 'Sadness'
        },
        {
            'text': 'I am feeling anxious about the exam.',
            'emotions': {
                'Joy': 0.02, 'Sadness': 0.1, 'Anger': 0.05, 'Fear': 0.7, 'Surprise': 0.03, 'Neutral': 0.1
            },
            'dominant_emotion': 'Fear'
        },
        {
            'text': 'What a surprise!',
            'emotions': {
                'Joy': 0.1, 'Sadness': 0.05, 'Anger': 0.05, 'Fear': 0.1, 'Surprise': 0.65, 'Neutral': 0.05
            },
            'dominant_emotion': 'Surprise'
        }
    ]


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_path(sample_dataset, test_data_dir):
    """Create a sample CSV file for testing."""
    csv_path = test_data_dir / "sample_data.csv"
    sample_dataset.to_csv(csv_path, index=False)
    return csv_path

