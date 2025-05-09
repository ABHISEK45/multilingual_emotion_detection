"""
Tests for the main module and CLI functionality.

This module tests the command-line interface and main functionality
of the multilingual emotion detection system.
"""

import os
import sys
import json
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.main import parse_arguments, validate_args, detect_emotion, batch_process, main, _save_results


class TestArgumentParsing:
    """Tests for command line argument parsing and validation."""
    
    def test_parse_arguments_with_text(self, monkeypatch):
        """Test parsing arguments with text argument."""
        # Mock sys.argv
        monkeypatch.setattr('sys.argv', ['main.py', '--text', 'Test text'])
        
        # Parse arguments
        args = parse_arguments()
        
        # Check arguments
        assert args.text == 'Test text'
        assert args.language == 'auto'  # Default value
        assert args.dataset is None
        assert args.model_path is not None  # Default value should be set
    
    def test_parse_arguments_with_dataset(self, monkeypatch):
        """Test parsing arguments with dataset argument."""
        # Mock sys.argv
        monkeypatch.setattr('sys.argv', ['main.py', '--dataset', 'data.csv', '--output', 'results.csv'])
        
        # Parse arguments
        args = parse_arguments()
        
        # Check arguments
        assert args.dataset == 'data.csv'
        assert args.output == 'results.csv'
        assert args.text is None
        assert args.batch_size == 16  # Default value
    
    def test_validate_args_text(self):
        """Test validation of text arguments."""
        # Create a mock args object with text
        args = MagicMock()
        args.text = "Test text"
        args.dataset = None
        args.language = "en"
        args.model_path = "models/test_model"
        args.output = None
        args.batch_size = 16
        
        # Validate arguments
        is_valid, _ = validate_args(args)
        
        # Should be valid
        assert is_valid
    
    def test_validate_args_dataset(self, tmp_path):
        """Test validation of dataset arguments."""
        # Create a sample dataset file
        dataset_path = tmp_path / "test_dataset.csv"
        pd.DataFrame({'text': ['Test']}).to_csv(dataset_path, index=False)
        
        # Create a mock args object with dataset
        args = MagicMock()
        args.text = None
        args.dataset = str(dataset_path)
        args.language = "auto"
        args.model_path = "models/test_model"
        args.output = str(tmp_path / "results.csv")
        args.batch_size = 16
        
        # Validate arguments
        is_valid, _ = validate_args(args)
        
        # Should be valid
        assert is_valid
    
    def test_validate_args_invalid_no_input(self):
        """Test validation with no input."""
        # Create a mock args object with no text or dataset
        args = MagicMock()
        args.text = None
        args.dataset = None
        
        # Validate arguments
        is_valid, error_message = validate_args(args)
        
        # Should be invalid
        assert not is_valid
        assert "must be provided" in error_message.lower()
    
    def test_validate_args_invalid_language(self):
        """Test validation with invalid language."""
        # Create a mock args object with invalid language
        args = MagicMock()
        args.text = "Test text"
        args.dataset = None
        args.language = "fr"  # Not supported
        
        # Validate arguments
        is_valid, error_message = validate_args(args)
        
        # Should be invalid
        assert not is_valid
        assert "invalid language" in error_message.lower()
    
    def test_validate_args_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        # Create a mock args object with invalid batch size
        args = MagicMock()
        args.text = None
        args.dataset = "data.csv"
        args.batch_size = 0  # Invalid
        
        # Validate arguments
        is_valid, error_message = validate_args(args)
        
        # Should be invalid
        assert not is_valid
        assert "batch size" in error_message.lower()
    
    def test_validate_args_nonexistent_dataset(self):
        """Test validation with nonexistent dataset file."""
        # Create a mock args object with nonexistent dataset
        args = MagicMock()
        args.text = None
        args.dataset = "nonexistent_file.csv"
        args.batch_size = 16
        
        # Validate arguments
        is_valid, error_message = validate_args(args)
        
        # Should be invalid
        assert not is_valid
        assert "not found" in error_message.lower()


class TestSingleTextProcessing:
    """Tests for processing a single text input."""
    
    def test_detect_emotion(self, monkeypatch):
        """Test emotion detection for a single text."""
        # Mock the EmotionDetectionModel
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            'text': 'Test text',
            'emotions': {'Joy': 0.8, 'Sadness': 0.1, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
            'dominant_emotion': 'Joy'
        }
        
        # Mock the model.EmotionDetectionModel constructor
        monkeypatch.setattr('src.model.EmotionDetectionModel', lambda *args, **kwargs: mock_model)
        
        # Mock preprocessor
        monkeypatch.setattr('src.preprocessor.detect_language', lambda text: 'en')
        monkeypatch.setattr('src.preprocessor.preprocess_text', lambda text, language: text.lower())
        
        # Call the function
        result = detect_emotion('Test text', 'auto', 'models/test_model')
        
        # Check the result
        assert result['text'] == 'Test text'
        assert result['dominant_emotion'] == 'Joy'
        assert result['emotions']['Joy'] == 0.8
        
        # Check that the model was called with the preprocessed text
        mock_model.predict.assert_called_once_with('test text')
    
    def test_detect_emotion_with_explicit_language(self, monkeypatch):
        """Test emotion detection with explicitly specified language."""
        # Mock the EmotionDetectionModel
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            'text': 'Test text',
            'emotions': {'Joy': 0.8, 'Sadness': 0.1, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
            'dominant_emotion': 'Joy'
        }
        
        # Mock the model.EmotionDetectionModel constructor
        monkeypatch.setattr('src.model.EmotionDetectionModel', lambda *args, **kwargs: mock_model)
        
        # Mock preprocessor for Hindi
        monkeypatch.setattr('src.preprocessor.preprocess_text', lambda text, language: f"preprocessed_{language}_{text}")
        
        # Call the function with explicit Hindi language
        result = detect_emotion('Test text', 'hi', 'models/test_model')
        
        # Check the result
        assert result['text'] == 'Test text'
        assert result['dominant_emotion'] == 'Joy'
        
        # Check that the model was called with the Hindi-preprocessed text
        mock_model.predict.assert_called_once_with('preprocessed_hi_Test text')
    
    def test_detect_emotion_error_handling(self, monkeypatch):
        """Test error handling in emotion detection."""
        # Mock preprocessor to raise an error
        def mock_preprocess_error(text, language):
            raise ValueError("Test preprocessing error")
        
        monkeypatch.setattr('src.preprocessor.detect_language', lambda text: 'en')
        monkeypatch.setattr('src.preprocessor.preprocess_text', mock_preprocess_error)
        
        # Call the function and expect an error
        with pytest.raises(RuntimeError):
            detect_emotion('Test text', 'auto', 'models/test_model')
    
    def test_detect_emotion_with_invalid_text(self, monkeypatch):
        """Test emotion detection with invalid text."""
        # Call the function with invalid text
        with pytest.raises(ValueError):
            detect_emotion('', 'auto', 'models/test_model')
        
        with pytest.raises(ValueError):
            detect_emotion(None, 'auto', 'models/test_model')


class TestBatchProcessing:
    """Tests for batch processing functionality."""
    
    def test_batch_process(self, monkeypatch, tmp_path, sample_dataset, sample_dataset_path):
        """Test basic batch processing."""
        # Mock the EmotionDetectionModel
        mock_model = MagicMock()
        mock_model.predict_batch.return_value = [
            {
                'text': 'Text 1',
                'emotions': {'Joy': 0.8, 'Sadness': 0.1, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
                'dominant_emotion': 'Joy'
            },
            {
                'text': 'Text 2',
                'emotions': {'Joy': 0.1, 'Sadness': 0.8, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
                'dominant_emotion': 'Sadness'
            }
        ]
        
        # Mock the model.EmotionDetectionModel constructor
        monkeypatch.setattr('src.model.EmotionDetectionModel', lambda *args, **kwargs: mock_model)
        
        # Mock evaluation module
        mock_evaluate = MagicMock()
        mock_evaluate.return_value = {'accuracy': 0.85, 'f1_score': 0.82}
        monkeypatch.setattr('src.evaluation.evaluate', mock_evaluate)
        
        # Create output path
        output_path = str(tmp_path / "results.json")
        
        # Call the function
        metrics = batch_process(
            dataset_path=sample_dataset_path,
            output_path=output_path,
            model_path='models/test_model',
            batch_size=2,
            show_progress=True
        )
        
        # Check the metrics
        assert metrics['accuracy'] == 0.85
        assert metrics['f1_score'] == 0.82
        
        # Check that the output file exists
        assert os.path.exists(output_path)
        
        # Check that the model was called with correct batch size
        mock_model.predict_batch.assert_called_once()
    
    def test_batch_process_error_handling(self, monkeypatch, sample_dataset_path):
        """Test error handling in batch processing."""
        # Mock preprocessor.load_dataset to raise an error
        def mock_load_dataset_error(path):
            raise ValueError("Test dataset loading error")
        
        monkeypatch.setattr('src.preprocessor.load_dataset', mock_load_dataset_error)
        
        # Call the function and expect an error
        with pytest.raises(ValueError):
            batch_process(
                dataset_path=sample_dataset_path,
                output_path=None,
                model_path='models/test_model',
                batch_size=2
            )
    
    def test_save_results(self, tmp_path):
        """Test saving results to different formats."""
        # Create sample results
        results = [
            {
                'text': 'Text 1',
                'emotions': {'Joy': 0.8, 'Sadness': 0.1, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
                'dominant_emotion': 'Joy'
            },
            {
                'text': 'Text 2',
                'emotions': {'Joy': 0.1, 'Sadness': 0.8, 'Anger': 0.05, 'Fear': 0.02, 'Surprise': 0.02, 'Neutral': 0.01},
                'dominant_emotion': 'Sadness'
            }
        ]
        
        # Test JSON output
        json_path = tmp_path / "results.json"
        _save_results(results, str(json_path))
        
        # Check that file exists and contains the expected data
        assert json_path.exists()
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
            assert len(loaded_data) == 2
            assert loaded_data[0]['dominant_emotion'] == 'Joy'
            assert loaded_data[1]['dominant_emotion'] == 'Sadness'
        
        # Test CSV output
        csv_path = tmp_path / "results.csv"
        _save_results(results, str(csv_path))
        
        # Check that file exists and contains the expected data
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df['dominant_emotion'][0] == 'Joy'
        assert df['dominant_emotion'][1] == 'Sadness'
        assert 'emotion_Joy' in df.columns
    
    def test_batch_process_memory_management(self, monkeypatch, sample_dataset_path, tmp_path):
        """Test memory management during batch processing."""
        # Mock gc.collect and torch.cuda.empty_cache to check if they're called
        mock_gc_collect = MagicMock()
        mock_cuda_empty_cache = MagicMock()
        
        monkeypatch.setattr('gc.collect', mock_gc_collect)
        monkeypatch.setattr('torch.cuda.empty_cache', mock_cuda_empty_cache)
        
        # Mock other dependencies to avoid actual processing
        mock_model = MagicMock()
        mock_model.predict_batch.return_value = []
        monkeypatch.setattr('src.model.EmotionDetectionModel', lambda *args, **kwargs: mock_model)
        
        monkeypatch.setattr('src.preprocessor.load_dataset', lambda path: pd.DataFrame({'text': []}))
        monkeypatch.setattr('src.evaluation.evaluate', lambda *args: {})
        
        # Call the function
        batch_process(
            dataset_path=sample_dataset_path,
            output_path=str(tmp_path / "results.json"),
            batch_size=2
        )
        
        # Check that memory cleanup was performed
        assert mock_gc_collect.called, "gc.collect should be called for memory cleanup"
        
        # cuda.empty_cache might not be called if CUDA

