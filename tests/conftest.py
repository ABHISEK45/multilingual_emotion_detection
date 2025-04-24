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

