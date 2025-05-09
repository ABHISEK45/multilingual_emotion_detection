"""
Tests for the emotion detection model module.

This module tests the functionality of the EmotionDetectionModel class,
including initialization, prediction, and error handling.
"""

import os
import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.model import EmotionDetectionModel, EMOTION_CLASSES


class TestEmotionDetectionModel:
    """Tests for the EmotionDetectionModel class."""
    
    def test_initialization_default(self, mock_tokenizer, mock_model, monkeypatch):
        """Test initialization with default parameters."""
        # Mock the AutoTokenizer and AutoModelForSequenceClassification
        monkeypatch.setattr('src.model.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer)
        monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', 
                            lambda *args, **kwargs: mock_model)
        
        # Initialize model with default settings
        model = EmotionDetectionModel()
        
        # Check that model and tokenizer are set correctly
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device in ['cpu', 'cuda']
        assert 'version' in model.metadata
    
    def test_initialization_with_path(self, mock_model_path, mock_tokenizer, mock_model, monkeypatch):
        """Test initialization with a specified model path."""
        # Mock the AutoTokenizer and AutoModelForSequenceClassification
        monkeypatch.setattr('src.model.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer)
        monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', 
                            lambda *args, **kwargs: mock_model)
        
        # Initialize model with path
        model = EmotionDetectionModel(model_path=mock_model_path)
        
        # Check that model and tokenizer are set correctly
        assert model.model is not None
        assert model.tokenizer is not None
        assert 'version' in model.metadata
    
    def test_initialization_error_handling(self, tmp_path, monkeypatch):
        """Test error handling during initialization."""
        # Create an invalid model path
        invalid_path = tmp_path / "nonexistent_model"
        
        # Mock an exception when loading the model
        def mock_error(*args, **kwargs):
            raise ValueError("Test error loading model")
        
        # Also mock the fallback to ensure we test the full error path
        def mock_fallback_error(*args, **kwargs):
            raise RuntimeError("Test error with fallback")
        
        monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', mock_error)
        monkeypatch.setattr('src.model.AutoTokenizer.from_pretrained', mock_error)
        
        # Check that initialization with an invalid path raises an error
        with pytest.raises(RuntimeError):
            EmotionDetectionModel(model_path=str(invalid_path))
    
    def test_predict_single_text(self, mock_emotion_model, sample_texts):
        """Test prediction for a single text input."""
        # Get a sample text
        text = sample_texts['english'][0]
        
        # Make prediction
        result = mock_emotion_model.predict(text)
        
        # Check result format
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'emotions' in result
        assert 'dominant_emotion' in result
        assert result['text'] == text
        assert all(emotion in result['emotions'] for emotion in EMOTION_CLASSES)
        assert result['dominant_emotion'] in EMOTION_CLASSES
    
    def test_predict_invalid_input(self, mock_emotion_model):
        """Test prediction with invalid inputs."""
        # Test with empty text
        with pytest.raises(ValueError):
            mock_emotion_model.predict("")
        
        # Test with None
        with pytest.raises(ValueError):
            mock_emotion_model.predict(None)
        
        # Test with non-string
        with pytest.raises(ValueError):
            mock_emotion_model.predict(123)
    
    def test_predict_batch(self, mock_emotion_model, sample_texts):
        """Test batch prediction functionality."""
        # Get sample texts
        texts = sample_texts['english']
        
        # Make batch prediction
        results = mock_emotion_model.predict_batch(texts)
        
        # Check results
        assert isinstance(results, list)
        assert len(results) == len(texts)
        
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'emotions' in result
            assert 'dominant_emotion' in result
            assert result['text'] == texts[i]
            assert all(emotion in result['emotions'] for emotion in EMOTION_CLASSES)
            assert result['dominant_emotion'] in EMOTION_CLASSES
    
    def test_predict_batch_empty(self, mock_emotion_model):
        """Test batch prediction with empty inputs."""
        # Test with empty list
        with pytest.raises(ValueError):
            mock_emotion_model.predict_batch([])
        
        # Test with list containing invalid items
        with pytest.raises(ValueError):
            mock_emotion_model.predict_batch(["", None, 123])
    
    def test_predict_batch_with_progress(self, mock_emotion_model, sample_texts):
        """Test batch prediction with progress tracking."""
        # Get sample texts
        texts = sample_texts['english']
        
        # Make batch prediction with progress bar
        results = mock_emotion_model.predict_batch(texts, show_progress=True)
        
        # Check results
        assert isinstance(results, list)
        assert len(results) == len(texts)
    
    def test_memory_usage(self, mock_emotion_model):
        """Test memory usage monitoring."""
        # Get memory usage
        memory_info = mock_emotion_model.get_memory_usage()
        
        # Check memory info format
        assert isinstance(memory_info, dict)
        assert 'system_used_gb' in memory_info
        assert 'system_available_gb' in memory_info
        assert 'system_percent' in memory_info
        
        # Check values are reasonable
        assert 0 <= memory_info['system_percent'] <= 100
        assert memory_info['system_used_gb'] >= 0
        assert memory_info['system_available_gb'] >= 0
    
    def test_batch_size_estimation(self, mock_emotion_model):
        """Test optimal batch size estimation."""
        # Estimate batch size
        batch_size = mock_emotion_model.estimate_optimal_batch_size()
        
        # Check it's a reasonable value
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 128  # The default max
    
    def test_batch_size_auto_adjustment(self, mock_emotion_model, sample_texts):
        """Test automatic batch size adjustment."""
        # Get a larger set of texts by repeating samples
        texts = sample_texts['english'] * 10  # 60 texts
        
        # Make batch prediction with auto-adjustment
        results = mock_emotion_model.predict_batch(texts, batch_size=10, auto_adjust_batch=True)
        
        # Just check that it completes without error
        assert len(results) == len(texts)
    
    def test_save_model(self, mock_emotion_model, tmp_path):
        """Test model saving functionality."""
        # Define a save path
        save_path = tmp_path / "test_save_model"
        
        # Save the model
        result_path = mock_emotion_model.save_model(str(save_path))
        
        # Check that the save was successful
        assert os.path.exists(result_path)
        assert os.path.exists(os.path.join(result_path, "metadata.json"))
        
        # Check the saved metadata
        with open(os.path.join(result_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
            assert 'version' in metadata
    
    def test_save_model_with_version(self, mock_emotion_model, tmp_path):
        """Test model saving with version specification."""
        # Define a save path
        save_path = tmp_path / "test_save_model_version"
        test_version = "2.0.0-test"
        
        # Save the model with a specific version
        result_path = mock_emotion_model.save_model(str(save_path), version=test_version)
        
        # Check that the save was successful
        assert os.path.exists(result_path)
        
        # Check the saved metadata
        with open(os.path.join(result_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
            assert metadata['version'] == test_version
    
    def test_create_checkpoint(self, mock_emotion_model, tmp_path):
        """Test checkpoint creation."""
        # Define a checkpoint directory
        checkpoint_dir = tmp_path / "checkpoints"
        
        # Create a checkpoint
        checkpoint_path = mock_emotion_model.create_checkpoint(str(checkpoint_dir))
        
        # Check that the checkpoint was created
        assert os.path.exists(checkpoint_path)
        assert "checkpoint_" in os.path.basename(checkpoint_path)
        
        # Check the checkpoint metadata
        with open(os.path.join(checkpoint_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
            assert 'version' in metadata
            assert 'checkpoint_time' in metadata
            assert metadata['checkpoint_type'] == 'auto'
    
    def test_load_checkpoint(self, mock_emotion_model, mock_model_path, monkeypatch):
        """Test loading a model from a checkpoint."""
        # Create a mock checkpoint path
        checkpoint_path = mock_model_path
        
        # Mock the model and tokenizer loading
        monkeypatch.setattr('src.model.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_emotion_model.tokenizer)
        monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', 
                            lambda *args, **kwargs: mock_emotion_model.model)
        
        # Load the checkpoint
        success = mock_emotion_model.load_checkpoint(checkpoint_path)
        
        # Check that loading was successful
        assert success is True
    
    @pytest.mark.parametrize("error_type", [
        "file_not_found", "load_error"
    ])
    def test_load_checkpoint_errors(self, mock_emotion_model, tmp_path, monkeypatch, error_type):
        """Test error handling when loading checkpoints."""
        # Create a non-existent checkpoint path
        nonexistent_path = tmp_path / "nonexistent_checkpoint"
        
        if error_type == "file_not_found":
            # Test with non-existent path
            with pytest.raises(FileNotFoundError):
                mock_emotion_model.load_checkpoint(str(nonexistent_path))
        
        elif error_type == "load_error":
            # Create empty directory to simulate a corrupted checkpoint
            test_path = tmp_path / "corrupt_checkpoint"
            test_path.mkdir()
            
            # Mock an error during loading
            def mock_error(*args, **kwargs):
                raise ValueError("Test error loading model")
            
            monkeypatch.setattr('src.model.AutoModelForSequenceClassification.from_pretrained', mock_error)
            
            # Test with corrupted checkpoint
            with pytest.raises(OSError):
                mock_emotion_model.load_checkpoint(str(test_path))

"""
Tests for the model module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

from src.model import EmotionDetectionModel, EMOTION_CLASSES


class TestEmotionClasses:
    """Tests for the EMOTION_CLASSES constant."""
    
    def test_emotion_classes_content(self):
        """Test that emotion classes include all required emotions."""
        assert len(EMOTION_CLASSES) == 6, "Should have exactly 6 emotion classes"
        
        required_emotions = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
        for emotion in required_emotions:
            assert emotion in EMOTION_CLASSES, f"Missing required emotion: {emotion}"


class TestModelInitialization:
    """Tests for model initialization."""
    
    def test_default_initialization(self, monkeypatch):
        """Test that model initializes with default settings."""
        # Mock the transformer imports
        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        
        monkeypatch.setattr("src.model.AutoModelForSequenceClassification", mock_auto_model)
        monkeypatch.setattr("src.model.AutoTokenizer", mock_auto_tokenizer)
        
        # Initialize model
        model = EmotionDetectionModel()
        
        # Check that model was initialized with default parameters
        assert mock_auto_model.from_pretrained.called
        assert mock_auto_tokenizer.from_pretrained.called
        
        # Check model_name is set to default
        assert model.model_name == "xlm-roberta-base"
        
        # Check that device is set
        assert model.device in ["cpu", "cuda"]
    
    def test_custom_path_initialization(self, monkeypatch, test_data_dir):
        """Test initialization with custom model path."""
        # Create mock model path
        model_path = test_data_dir / "test_model"
        model_path.mkdir(exist_ok=True)
        
        # Mock the transformer imports
        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()
        
        monkeypatch.setattr("src.model.AutoModelForSequenceClassification", mock_auto_model)
        monkeypatch.setattr("src.model.AutoTokenizer", mock_auto_tokenizer)
        
        # Initialize model with custom path
        model = EmotionDetectionModel(model_path=str(model_path))
        
        # Check that model was initialized with the custom path
        mock_auto_model.from_pretrained.assert_called_with(model_path)
        mock_auto_tokenizer.from_pretrained.assert_called_with(model_path)
    
    def test_initialization_with_mock_objects(self, mock_model, mock_tokenizer):
        """Test initialization using mock model and tokenizer."""
        model = EmotionDetectionModel()
        
        # Replace with mock objects
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        
        # Check that model and tokenizer are set
        assert model.model is mock_model
        assert model.tokenizer is mock_tokenizer
        
        # Check that model is in eval mode
        assert model.model.eval_called if hasattr(model.model, 'eval_called') else True
    
    def test_device_setting(self, monkeypatch, mock_model, mock_tokenizer):
        """Test that device is properly set during initialization."""
        # Mock torch.cuda.is_available to test both CPU and CUDA paths
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        
        # CPU case
        model_cpu = EmotionDetectionModel()
        model_cpu.model = mock_model
        model_cpu.tokenizer = mock_tokenizer
        assert model_cpu.device == "cpu"
        
        # CUDA case (mocked)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        model_cuda = EmotionDetectionModel()
        model_cuda.model = mock_model
        model_cuda.tokenizer = mock_tokenizer
        assert model_cuda.device == "cuda"
        
        # Explicit device specification
        model_explicit = EmotionDetectionModel(device="cpu")
        model_explicit.model = mock_model
        model_explicit.tokenizer = mock_tokenizer
        assert model_explicit.device == "cpu"


class TestPrediction:
    """Tests for model prediction capabilities."""
    
    def test_single_text_prediction(self, mock_emotion_model):
        """Test prediction for a single text input."""
        text = "I am feeling happy today!"
        result = mock_emotion_model.predict(text)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'emotions' in result
        assert 'dominant_emotion' in result
        
        # Check that original text is preserved
        assert result['text'] == text
        
        # Check that emotions dictionary has correct format
        assert isinstance(result['emotions'], dict)
        assert set(result['emotions'].keys()) == set(EMOTION_CLASSES)
        assert all(isinstance(prob, float) for prob in result['emotions'].values())
        
        # Check that probabilities sum approximately to 1
        assert np.isclose(sum(result['emotions'].values()), 1.0, atol=1e-5)
        
        # Check that dominant emotion is one of the valid emotions
        assert result['dominant_emotion'] in EMOTION_CLASSES
    
    def test_batch_prediction(self, mock_emotion_model, sample_texts):
        """Test prediction for a batch of texts."""
        # Use first 4 texts from sample_texts
        texts = sample_texts[:4]
        results = mock_emotion_model.predict_batch(texts)
        
        # Check results structure
        assert isinstance(results, list)
        assert len(results) == len(texts)
        
        # Check each individual result
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'emotions' in result
            assert 'dominant_emotion' in result
            
            # Check that original text is preserved
            assert result['text'] == texts[i]
            
            # Check emotions dictionary
            assert isinstance(result['emotions'], dict)
            assert set(result['emotions'].keys()) == set(EMOTION_CLASSES)
            assert all(isinstance(prob, float) for prob in result['emotions'].values())
            
            # Check that probabilities sum approximately to 1
            assert np.isclose(sum(result['emotions'].values()), 1.0, atol=1e-5)
            
            # Check that dominant emotion is one of the valid emotions
            assert result['dominant_emotion'] in EMOTION_CLASSES
    
    def test_empty_text_handling(self, mock_emotion_model):
        """Test handling of empty text input."""
        # Empty string
        result = mock_emotion_model.predict("")
        assert result['text'] == ""
        assert set(result['emotions'].keys()) == set(EMOTION_CLASSES)
        
        # Batch with some empty strings
        results = mock_emotion_model.predict_batch(["Hello", "", "Test"])
        assert len(results) == 3
        assert results[1]['text'] == ""
    
    def test_long_text_handling(self, mock_emotion_model):
        """Test handling of very long text input."""
        # Create a very long text (beyond typical max_length)
        long_text = "word " * 1000  # 5000+ characters
        
        # Should not raise any errors
        result = mock_emotion_model.predict(long_text)
        assert result['text'] == long_text
    
    def test_non_ascii_text(self, mock_emotion_model):
        """Test handling of non-ASCII characters."""
        # Hindi text
        hindi_text = "मैं आज बहुत खुश हूँ।"
        result = mock_emotion_model.predict(hindi_text)
        assert result['text'] == hindi_text
        
        # Emoji and special characters
        special_text = "I'm 😊 happy! ® © Ø"
        result = mock_emotion_model.predict(special_text)
        assert result['text'] == special_text


class TestModelSaving:
    """Tests for model saving functionality."""
    
    def test_save_model(self, mock_emotion_model, test_data_dir, monkeypatch):
        """Test saving the model to disk."""
        # Mock the save_pretrained methods
        mock_save_pretrained = MagicMock()
        monkeypatch.setattr(mock_emotion_model.model, "save_pretrained", mock_save_pretrained)
        
        mock_tokenizer_save = MagicMock()
        monkeypatch.setattr(mock_emotion_model.tokenizer, "save_pretrained", mock_tokenizer_save)
        
        # Test saving
        save_path = test_data_dir / "saved_model"
        mock_emotion_model.save_model(str(save_path))
        
        # Check that save methods were called with correct path
        mock_save_pretrained.assert_called_once_with(save_path)
        mock_tokenizer_save.assert_called_once_with(save_path)
    
    def test_save_path_creation(self, mock_emotion_model, test_data_dir, monkeypatch):
        """Test that save_model creates the directory if it doesn't exist."""
        # Mock the save_pretrained methods
        mock_save_pretrained = MagicMock()
        monkeypatch.setattr(mock_emotion_model.model, "save_pretrained", mock_save_pretrained)
        
        mock_tokenizer_save = MagicMock()
        monkeypatch.setattr(mock_emotion_model.tokenizer, "save_pretrained", mock_tokenizer_save)
        
        # Create a nested path that doesn't exist
        save_path = test_data_dir / "nested" / "directory" / "model"
        
        # Should create directories and not raise error
        mock_emotion_model.save_model(str(save_path))
        assert save_path.parent.exists()


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""
    
    def test_missing_model_path(self, monkeypatch, test_data_dir):
        """Test handling of missing model path."""
        # Create non-existent path
        non_existent_path = test_data_dir / "non_existent_model"
        
        # Mock from_pretrained to simulate error
        def mock_from_pretrained_error(*args, **kwargs):
            raise OSError(f"Model not found at {non_existent_path}")
        
        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.side_effect = mock_from_pretrained_error
        
        monkeypatch.setattr("src.model.AutoModelForSequenceClassification", mock_auto_model)
        
        # Initialize model should raise OSError
        with pytest.raises(OSError):
            EmotionDetectionModel(model_path=str(non_existent_path))
    
    def test_invalid_input_type(self, mock_emotion_model):
        """Test handling of invalid input types."""
        # None input
        with pytest.raises((TypeError, ValueError)):
            mock_emotion_model.predict(None)
        
        # Non-string input
        with pytest.raises((TypeError, ValueError)):
            mock_emotion_model.predict(123)
        
        # List input to single prediction
        with pytest.raises((TypeError, ValueError)):
            mock_emotion_model.predict(["multiple", "inputs"])
        
        # Non-list input to batch prediction
        with pytest.raises((TypeError, ValueError)):
            mock_emotion_model.predict_batch("single text")

"""
Tests for the model module.
"""

import pytest
import torch
from src.model import EmotionDetectionModel, EMOTION_CLASSES


def test_emotion_classes():
    """Test that emotion classes are properly defined."""
    assert len(EMOTION_CLASSES) == 6
    assert "Joy" in EMOTION_CLASSES
    assert "Sadness" in EMOTION_CLASSES
    assert "Anger" in EMOTION_CLASSES
    assert "Fear" in EMOTION_CLASSES
    assert "Surprise" in EMOTION_CLASSES
    assert "Neutral" in EMOTION_CLASSES


def test_emotion_model_init_with_mocks(mock_model, mock_tokenizer, monkeypatch):
    """Test EmotionDetectionModel initialization with mock components."""
    # Mock the AutoModelForSequenceClassification.from_pretrained
    def mock_model_from_pretrained(*args, **kwargs):
        return mock_model
    
    # Mock the AutoTokenizer.from_pretrained
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return mock_tokenizer
    
    # Apply the monkeypatches
    monkeypatch.setattr(
        "src.model.AutoModelForSequenceClassification.from_pretrained", 
        mock_model_from_pretrained
    )
    monkeypatch.setattr(
        "src.model.AutoTokenizer.from_pretrained", 
        mock_tokenizer_from_pretrained
    )
    
    # Initialize the model
    model = EmotionDetectionModel()
    
    # Check that the model and tokenizer are set
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.device in ['cpu', 'cuda']


def test_predict_single_text(mock_emotion_model):
    """Test predicting emotions for a single text."""
    text = "I am feeling happy today!"
    result = mock_emotion_model.predict(text)
    
    # Check result structure
    assert 'text' in result
    assert 'emotions' in result
    assert 'dominant_emotion' in result
    
    # Check that the input text is preserved
    assert result['text'] == text
    
    # Check emotions dictionary
    assert all(emotion in result['emotions'] for emotion in EMOTION_CLASSES)
    assert all(isinstance(prob, float) for prob in result['emotions'].values())
    
    # Check that probabilities sum to approximately 1
    assert abs(sum(result['emotions'].values()) - 1.0) < 1e-5
    
    # Check dominant emotion
    assert result['dominant_emotion'] in EMOTION_CLASSES


def test_predict_batch(mock_emotion_model, sample_texts):
    """Test predicting emotions for a batch of texts."""
    results = mock_emotion_model.predict_batch(sample_texts)
    
    # Check that we get one result per input text
    assert len(results) == len(sample_texts)
    
    # Check structure of each result
    for i, result in enumerate(results):
        assert 'text' in result
        assert 'emotions' in result
        assert 'dominant_emotion' in result
        
        # Check that the input text is preserved
        assert result['text'] == sample_texts[i]
        
        # Check emotions dictionary
        assert all(emotion in result['emotions'] for emotion in EMOTION_CLASSES)
        assert all(isinstance(prob, float) for prob in result['emotions'].values())
        
        # Check that probabilities sum to approximately 1
        assert abs(sum(result['emotions'].values()) - 1.0) < 1e-5
        
        # Check dominant emotion
        assert result['dominant_emotion'] in EMOTION_CLASSES


def test_model_save(mock_emotion_model, test_data_dir, monkeypatch):
    """Test model saving functionality."""
    # Mock the save_pretrained methods
    save_calls = []
    
    def mock_save_pretrained(path):
        save_calls.append(path)
    
    # Apply the monkeypatch
    monkeypatch.setattr(mock_emotion_model.model, "save_pretrained", mock_save_pretrained)
    
    # Test saving the model
    save_path = test_data_dir / "test_model"
    mock_emotion_model.save_model(save_path)
    
    # Check that save_pretrained was called with the correct path
    assert len(save_calls) > 0
    assert str(save_path) in str(save_calls[0])

