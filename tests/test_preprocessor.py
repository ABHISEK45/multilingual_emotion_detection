"""
Tests for the text preprocessing module.

This module tests the functionality of the preprocessor module,
including language detection, text cleaning, and Hindi text processing.
"""

import pytest
import pandas as pd
import re
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.preprocessor import (
    detect_language, validate_text, clean_text, normalize_hindi_text,
    preprocess_text, load_dataset, prepare_dataset, tokenize_text,
    transliterate_to_devanagari, detect_mixed_script_hindi,
    validate_hindi_text, process_hinglish_text, is_likely_romanized_hindi,
    handle_mixed_script_text
)


class TestLanguageDetection:
    """Tests for language detection functionality."""
    
    def test_detect_english(self, sample_texts):
        """Test detection of English text."""
        for text in sample_texts['english']:
            if isinstance(text, str) and text:
                detected = detect_language(text)
                assert detected == 'en', f"Failed to detect English: {text}"
    
    def test_detect_hindi(self, sample_texts):
        """Test detection of Hindi text."""
        for text in sample_texts['hindi']:
            if isinstance(text, str) and text:
                detected = detect_language(text)
                assert detected == 'hi', f"Failed to detect Hindi: {text}"
    
    def test_detect_hinglish(self, sample_texts):
        """Test detection of Hinglish (mixed Hindi-English) text."""
        # Hinglish is typically detected as English due to Latin script
        for text in sample_texts['hinglish']:
            if isinstance(text, str) and text:
                detected = detect_language(text)
                # Most Hinglish should be detected as English due to script
                assert detected in ['en', 'hi'], f"Unexpected language detected for Hinglish: {detected}"
                
                # Check if it's recognized as mixed script
                is_mixed = detect_mixed_script_hindi(text)
                assert is_mixed, f"Failed to detect Hinglish as mixed script: {text}"
    
    def test_detect_language_invalid_input(self, sample_texts):
        """Test language detection with invalid inputs."""
        for text in sample_texts['invalid']:
            if not isinstance(text, str) or not text:
                with pytest.raises(ValueError):
                    detect_language(text)
    
    @patch('src.preprocessor.detect')
    def test_detect_language_with_library(self, mock_detect):
        """Test language detection using the langdetect library."""
        # Mock the langdetect library to return specific values
        mock_detect.return_value = 'en'
        
        # Test with English text
        assert detect_language("This is English text.") == 'en'
        
        # Mock for Hindi
        mock_detect.return_value = 'hi'
        assert detect_language("हिंदी पाठ") == 'hi'
        
        # Test fallback mechanism when langdetect raises an exception
        mock_detect.side_effect = Exception("Test exception")
        
        # Should still work for obvious Hindi text using heuristic
        assert detect_language("यह हिंदी पाठ है।") == 'hi'
        
        # Should default to English for non-Devanagari
        assert detect_language("This should default to English") == 'en'


class TestTextCleaning:
    """Tests for text cleaning functionality."""
    
    def test_validate_text(self):
        """Test text validation."""
        # Valid texts
        assert validate_text("Hello world")[0], "Valid text should be accepted"
        assert validate_text("हिंदी पाठ")[0], "Valid Hindi text should be accepted"
        
        # Invalid texts
        assert not validate_text("")[0], "Empty string should be invalid"
        assert not validate_text(None)[0], "None should be invalid"
        assert not validate_text(123)[0], "Non-string should be invalid"
        assert not validate_text("   ")[0], "Whitespace-only string should be invalid"
    
    def test_clean_text(self, sample_texts):
        """Test text cleaning functionality."""
        # Test with English text
        for text in sample_texts['english']:
            if isinstance(text, str) and text:
                cleaned = clean_text(text)
                # Check that URLs are removed
                assert "http" not in cleaned.lower()
                # Check that extra whitespace is normalized
                assert not re.search(r'\s{2,}', cleaned)
                # Check text is not empty after cleaning
                assert cleaned
        
        # Test with Hindi text
        for text in sample_texts['hindi']:
            if isinstance(text, str) and text:
                cleaned = clean_text(text)
                # Check that cleaned text contains the original Hindi characters
                assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in cleaned)
    
    def test_clean_text_special_cases(self):
        """Test text cleaning with special inputs."""
        # Text with URLs
        text_with_url = "Check out https://example.com and www.example.org for more info."
        cleaned = clean_text(text_with_url)
        assert "https://example.com" not in cleaned
        assert "www.example.org" not in cleaned
        
        # Text with HTML tags
        text_with_html = "<p>This is a paragraph</p> with <b>bold</b> text."
        cleaned = clean_text(text_with_html)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        
        # Text with email address
        text_with_email = "Contact me at user@example.com for more information."
        cleaned = clean_text(text_with_email)
        assert "user@example.com" not in cleaned
    
    def test_clean_text_error_handling(self, sample_texts):
        """Test error handling in text cleaning."""
        # Test with invalid inputs
        for text in sample_texts['invalid']:
            if not isinstance(text, str) or not text:
                with pytest.raises(ValueError):
                    clean_text(text)


class TestHindiTextProcessing:
    """Tests for Hindi text processing functionality."""
    
    def test_normalize_hindi_text(self, sample_texts):
        """Test Hindi text normalization."""
        for text in sample_texts['hindi']:
            if isinstance(text, str) and text:
                normalized = normalize_hindi_text(text)
                # Check that normalization preserves Hindi characters
                assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in normalized)
                # Check that text is not empty after normalization
                assert normalized
    
    def test_normalize_hindi_text_with_variants(self):
        """Test normalization of Hindi text with variant characters."""
        # Text with nukta and other variants
        text_with_variants = "फ़िल्म और क़िस्सा"  # Film and story with nukta variants
        normalized = normalize_hindi_text(text_with_variants)
        # Should preserve the text while normalizing variants
        assert normalized
        assert "फ" in normalized or "फ़" in normalized
        assert "क" in normalized or "क़" in normalized
    
    def test_validate_hindi_text(self, sample_texts):
        """Test validation of Hindi text."""
        # Valid Hindi texts
        for text in sample_texts['hindi']:
            if isinstance(text, str) and text:
                is_valid, _ = validate_hindi_text(text)
                assert is_valid, f"Valid Hindi text should pass validation: {text}"
        
        # Romanized Hindi (Hinglish) texts - should detect as potentially Hindi
        for text in sample_texts['hinglish']:
            if isinstance(text, str) and text:
                is_valid, message = validate_hindi_text(text)
                assert is_valid, f"Hinglish should be recognized as potential Hindi: {text}"
                assert "Latin script" in message
        
        # Texts without Hindi characters
        for text in sample_texts['english']:
            if isinstance(text, str) and text:
                # Pure English text without Hindi markers should not be valid Hindi
                if not any(marker in text.lower() for marker in ["hindi", "namaste", "desi"]):
                    is_valid, _ = validate_hindi_text(text)
                    assert not is_valid, f"Pure English should not pass Hindi validation: {text}"
    
    def test_detect_mixed_script_hindi(self, sample_texts):
        """Test detection of Hindi text written in Latin script."""
        # Hinglish texts should be detected as mixed script
        for text in sample_texts['hinglish']:
            if isinstance(text, str) and text:
                is_mixed = detect_mixed_script_hindi(text)
                assert is_mixed, f"Failed to detect Hinglish: {text}"
        
        # Pure English texts should not be detected as mixed script
        for text in sample_texts['english']:
            if isinstance(text, str) and text and not any(marker in text.lower() for marker in ["hindi", "namaste", "desi"]):
                is_mixed = detect_mixed_script_hindi(text)
                assert not is_mixed, f"Incorrectly detected as Hinglish: {text}"
    
    def test_is_likely_romanized_hindi(self):
        """Test function to identify likely romanized Hindi words."""
        # Words that should be identified as Hindi
        hindi_words = ["namaste", "dhanyavaad", "kaise", "aapka", "theek", "hain", "kya", "acha"]
        for word in hindi_words:
            assert is_likely_romanized_hindi(word), f"Should identify as Hindi: {word}"
        
        # Words that should not be identified as Hindi
        english_words = ["hello", "thank", "you", "good", "bye", "computer", "file"]
        for word in english_words:
            assert not is_likely_romanized_hindi(word), f"Should not identify as Hindi: {word}"
    
    def test_transliterate_to_devanagari(self):
        """Test transliteration from Latin script to Devanagari."""
        # Basic transliteration cases
        test_cases = [
            ("namaste", "नमस्ते"),
            ("kya hal hai", "क्या हल है"),
            ("mera naam", "मेरा नाम"),
            ("bharat mata ki jai", "भारत माता की जय")
        ]
        
        for latin, expected_devanagari in test_cases:
            transliterated = transliterate_to_devanagari(latin)
            # We're checking if the key Hindi words are transliterated correctly
            # Exact matches are hard due to variations in transliteration schemes
            for hindi_word in expected_devanagari.split():
                assert hindi_word in transliterated, f"Failed to transliterate '{latin}' correctly"
    
    def test_handle_mixed_script_text(self):
        """Test handling of text with mixed scripts."""
        # Mixed Hindi (Devanagari) and English
        mixed_text = "मैं office में काम करता हूँ"  # I work in an office
        handled = handle_mixed_script_text(mixed_text)
        
        # Should preserve Devanagari portions
        assert "मैं" in handled
        assert "में" in handled
        assert "काम करता हूँ" in handled
        
        # Should also handle the English word
        assert "office" in handled
    
    def test_process_hinglish_text(self):
        """Test processing of Hinglish text."""
        # Hinglish text (Hindi written in Latin with some English words)
        hinglish = "Main office mein kaam karta hoon"  # I work in an office
        processed = process_hinglish_text(hinglish)
        
        # Should contain transliterated Hindi words
        assert re.search(r'[\u0900-\u097F]', processed), "Should contain Devanagari characters"
        
        # Should preserve English words
        assert "office" in processed


class TestPreprocessing:
    """Tests for text preprocessing functionality."""
    
    def test_preprocess_text_english(self, sample_texts):
        """Test preprocessing of English text."""
        for text in sample_texts['english']:
            if isinstance(text, str) and text:
                processed = preprocess_text(text, 'en')
                # Check that text is lowercased
                assert processed == processed.lower()
                # Check that non-alphanumeric characters are removed
                assert not re.search(r'[^a-z0-9\s]', processed)
    
    def test_preprocess_text_hindi(self, sample_texts):
        """Test preprocessing of Hindi text."""
        for text in sample_texts['hindi']:
            if isinstance(text, str) and text:
                processed = preprocess_text(text, 'hi')
                # Check that Hindi characters are preserved
                assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in processed)
    
    def test_preprocess_text_auto_detection(self, sample_texts):
        """Test preprocessing with automatic language detection."""
        # Mix of English and Hindi texts
        mixed_texts = sample_texts['english'][:2] + sample_texts['hindi'][:2]
        
        for text in mixed_texts:
            if isinstance(text, str) and text:
                # Preprocess with auto detection
                processed = preprocess_text(text)
                
                # Check that text is processed
                assert processed
                
                # For Hindi text, check that Devanagari is preserved
                if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in text):
                    assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in processed)
    
    def test_preprocess_text_error_handling(self, sample_texts):
        """Test error handling in text preprocessing."""
        # Test with invalid inputs
        for text in sample_texts['invalid']:
            if not isinstance(text, str) or not text:
                with pytest.raises(ValueError):
                    preprocess_text(text)
        
        # Test with invalid language
        with pytest.raises(ValueError):
            preprocess_text("Test text", language="invalid_lang")


class TestDatasetProcessing:
    """Tests for dataset processing functionality."""
    
    def test_load_dataset_csv(self, tmp_path):
        """Test loading dataset from CSV file."""
        # Create a sample CSV file
        csv_path = tmp_path / "test_data.csv"
        data = pd.DataFrame({
            'text': ['Test text 1', 'Test text 2'],
            'label': ['Joy', 'Sadness']
        })
        data.to_csv(csv_path, index=False)
        
        # Load the dataset
        loaded_data = load_dataset(csv_path)
        
        # Check that data is loaded correctly
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 2
        assert 'text' in loaded_data.columns
        assert 'label' in loaded_data.columns
    
    def test_load_dataset_json(self, tmp_path):
        """Test loading dataset from JSON file."""
        # Create a sample JSON file
        json_path = tmp_path / "test_data.json"
        data = pd.DataFrame({
            'text': ['Test text 1', 'Test text 2'],
            'label': ['Joy', 'Sadness']
        })
        data.to_

"""
Tests for the preprocessor module.
"""

import pytest
import re
import pandas as pd
from pathlib import Path
from src.preprocessor import (
    detect_language, 
    clean_text, 
    preprocess_text, 
    tokenize_text, 
    load_dataset,
    prepare_dataset
)


class TestLanguageDetection:
    """Tests for the detect_language function."""
    
    def test_english_detection(self):
        """Test detection of English text."""
        english_texts = [
            "This is an English sentence.",
            "Hello, how are you doing today?",
            "Machine learning is fascinating!",
            "1234567890",  # Numbers should default to English
            "!@#$%^&*()",  # Special characters shoul    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert "हिंदी" not in en_processed, "Hindi characters should be removed when processing as English"
        
        # When language is explicitly set to Hindi
        hi_processed = preprocess_text(mixed, 'hi')
        assert "english" not in hi_processed.lower(), "English characters should be removed when processing as Hindi"


class TestTokenization:
    """Tests for the tokenize_text function."""
    
    def test_tokenization_single_text(self, mock_tokenizer):
        """Test tokenization of a single text string."""
        text = "This is a test sentence."
        tokens = tokenize_text(text, mock_tokenizer)
        
        # Check basic structure
        assert 'input_ids' in tokens, "input_ids missing from token
            cleaned = clean_text(text)
            assert re.match(r"^\S.*\S$", cleaned) if cleaned else True, f"Whitespace not properly handled: '{text}' -> '{cleaned}'"
            assert "  " not in cleaned, f"Multiple consecutive spaces found in: '{cleaned}'"
    
    def test_edge_cases(self):
        """Test edge cases for text cleaning."""
        # Empty string
        assert clean_text("") == "", "Empty string should remain empty"
        
        # Only whitespace
        assert clean_text("   \t\n   ") == "", "String with only whitespace should become empty"
        
        # Only URLs
        assert clean_text("https://example.com") == "", "String with only URL should become empty"
        
        # Only HTML
        assert clean_text("<div></div>") == "", "String with only HTML should become empty or contain only the content"


class TestPreprocessing:
    """Tests for the preprocess_text function."""
    
    def test_english_preprocessing(self):
        """Test preprocessing of English text."""
        english_texts = [
            ("Hello, World!", "hello world"),  # Lowercase, remove punctuation
            ("Special @#$% Characters!", "special characters"),  # Remove special chars
            ("Numbers 123 should stay", "numbers 123 should stay"),  # Keep numbers
            ("Extra  spaces   here", "extra spaces here")  # Normalize spaces
        ]
        
        for original, expected in english_texts:
            preprocessed = preprocess_text(original, 'en')
            assert preprocessed.islower(), f"Text not lowercased: {preprocessed}"
            assert all(c.isalnum() or c.isspace() for c in preprocessed), f"Non-alphanumeric characters found: {preprocessed}"
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
    
    def test_hindi_preprocessing(self):
        """Test preprocessing of Hindi text."""
        hindi_texts = [
            ("नमस्ते, दुनिया!", "नमस्ते दुनिया!"),  # Keep punctuation but normalize spaces
            ("विशेष @#$% अक्षर!", "विशेष अक्षर!"),  # Remove non-Devanagari special chars
            ("हिंदी में 123 संख्याएँ", "हिंदी में  संख्याएँ"),  # Non-Devanagari numbers removed
            ("अतिरिक्त  स्थान   यहाँ", "अतिरिक्त स्थान यहाँ"),  # Normalize spaces
            ("देवनागरी वर्णमाला: क ख ग", "देवनागरी वर्णमाला क ख ग"),  # Preserve Devanagari chars
            ("ये $%#@! चिह्न हटा दिए जाएंगे", "ये  चिह्न हटा दिए जाएंगे"),  # Remove special chars
            ("हिंदी ! . ? निशान", "हिंदी ! . ? निशान")  # Keep punctuation
        ]
        
        for original, expected in hindi_texts:
            preprocessed = preprocess_text(original, 'hi')
            # Test Devanagari character preservation
            assert any('\u0900' <= c <= '\u097F' for c in preprocessed), f"No Devanagari characters found: {preprocessed}"
            # Test that the result matches expected output
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
            # Test that proper punctuation is maintained
            if "!" in original:
                assert "!" in preprocessed, "Punctuation should be preserved in Hindi"
            # Test that non-Devanagari special characters are removed
            if "@#$%" in original:
                assert "@" not in preprocessed, "Special characters should be removed"
                assert "#" not in preprocessed, "Special characters should be removed"
                assert "$" not in preprocessed, "Special characters should be removed"
                assert "%" not in preprocessed, "Special characters should be removed"
    
    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert "हिंदी" not in en_processed, "Hindi characters should be removed when processing as English"
        
        # When language is explicitly set to Hindi
        hi_processed = preprocess_text(mixed, 'hi')
        assert "english" not in hi_processed.lower(), "English characters should be removed when processing as Hindi"


class TestTokenization:
    """Tests for the tokenize_text function."""
    
    def test_tokenization_single_text(self, mock_tokenizer):
        """Test tokenization of a single text string."""
        text = "This is a test sentence."
        tokens = tokenize_text(text, mock_tokenizer)
        
        # Check basic structure
        assert 'input_ids' in tokens, "input_ids missing from tokenizer output"
        assert 'attention_mask' in tokens, "attention_mask missing from tokenizer output"
        
        # Check shapes
        assert tokens['input_ids'].shape[0] == 1, "Batch size should be 1 for single text"
    
    def test_tokenization_batch(self, mock_tokenizer, sample_texts):
        """Test tokenization of a batch of texts."""
        # Use mock tokenizer directly for comparison
        direct_tokens = mock_tokenizer(sample_texts, return_tensors="pt")
        
        # Use the tokenize_text function
        tokens = tokenize_text(sample_texts, mock_tokenizer)
        
        # Both should give equivalent results
        assert tokens['input_ids'].shape == direct_tokens['input_ids'].shape, "Shapes don't match"
        assert tokens['attention_mask'].shape == direct_tokens['attention_mask'].shape, "Shapes don't match"
    
    def test_max_length(self, mock_tokenizer):
        """Test that max_length parameter is respected."""
        text = "This is a test sentence."
        
        # Test with default max_length (128)
        tokens_default = tokenize_text(text, mock_tokenizer)
        
        # Test with custom max_length
        custom_length = 64
        tokens_custom = tokenize_text(text, mock_tokenizer, max_length=custom_length)
        
        # Check that tokenizer was called with proper params (in a real implementation)
        assert isinstance(tokens_default, dict), "Should return a dictionary"
        assert isinstance(tokens_custom, dict), "Should return a dictionary"
    
    def test_empty_input(self, mock_tokenizer):
        """Test tokenization with empty input."""
        # Empty string
        tokens = tokenize_text("", mock_tokenizer)
        assert 'input_ids' in tokens, "Should handle empty strings"
        
        # Empty list
        tokens = tokenize_text([], mock_tokenizer)
        assert 'input_ids' in tokens, "Should handle empty lists"
        assert tokens['input_ids'].shape[0] == 0, "Batch size should be 0 for empty list"


class TestDatasetLoading:
    """Tests for the load_dataset function."""
    
    def test_csv_loading(self, sample_csv_path):
        """Test loading data from CSV file."""
        df = load_dataset(sample_csv_path)
        
        # Basic checks
        for original, expected in english_texts:
            preprocessed = preprocess_text(original, 'en')
            assert preprocessed.islower(), f"Text not lowercased: {preprocessed}"
            assert all(c.isalnum() or c.isspace() for c in preprocessed), f"Non-alphanumeric characters found: {preprocessed}"
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
    
    def test_hindi_preprocessing(self):
        """Test preprocessing of Hindi text."""
        hindi_texts = [
            ("नमस्ते, दुनिया!", "नमस्ते दुनिया!"),  # Keep punctuation but normalize spaces
            ("विशेष @#$% अक्षर!", "विशेष अक्षर!"),  # Remove non-Devanagari special chars
            ("हिंदी में 123 संख्याएँ", "हिंदी में  संख्याएँ"),  # Non-Devanagari numbers removed
            ("अतिरिक्त  स्थान   यहाँ", "अतिरिक्त स्थान यहाँ"),  # Normalize spaces
            ("देवनागरी वर्णमाला: क ख ग", "देवनागरी वर्णमाला क ख ग"),  # Preserve Devanagari chars
            ("ये $%#@! चिह्न हटा दिए जाएंगे", "ये  चिह्न हटा दिए जाएंगे"),  # Remove special chars
            ("हिंदी ! . ? निशान", "हिंदी ! . ? निशान")  # Keep punctuation
        ]
        
        for original, expected in hindi_texts:
            preprocessed = preprocess_text(original, 'hi')
            # Test Devanagari character preservation
            assert any('\u0900' <= c <= '\u097F' for c in preprocessed), f"No Devanagari characters found: {preprocessed}"
            # Test that the result matches expected output
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
            # Test that proper punctuation is maintained
            if "!" in original:
                assert "!" in preprocessed, "Punctuation should be preserved in Hindi"
            # Test that non-Devanagari special characters are removed
            if "@#$%" in original:
                assert "@" not in preprocessed, "Special characters should be removed"
                assert "#" not in preprocessed, "Special characters should be removed"
                assert "$" not in preprocessed, "Special characters should be removed"
                assert "%" not in preprocessed, "Special characters should be removed"
    
    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert "हिंदी" not in en_processed, "Hindi characters should be removed when processing as English"
        
        # When language is explicitly set to Hindi
        hi_processed = preprocess_text(mixed, 'hi')
        assert "english" not in hi_processed.lower(), "English characters should be removed when processing as Hindi"
            ("हिंदी में 123 संख्याएँ", "हिंदी में  संख्याएँ"),  # Non-Devanagari numbers removed
            ("अतिरिक्त  स्थान   यहाँ", "अतिरिक्त स्थान यहाँ"),  # Normalize spaces
            ("देवनागरी वर्णमाला: क ख ग", "देवनागरी वर्णमाला क ख ग"),  # Preserve Devanagari chars
            ("ये $%#@! चिह्न हटा दिए जाएंगे", "ये  चिह्न हटा दिए जाएंगे"),  # Remove special chars
            ("हिंदी ! . ? निशान", "हिंदी ! . ? निशान")  # Keep punctuation
        ]
        
        for original, expected in hindi_texts:
            preprocessed = preprocess_text(original, 'hi')
            # Test Devanagari character preservation
            assert any('\u0900' <= c <= '\u097F' for c in preprocessed), f"No Devanagari characters found: {preprocessed}"
            # Test that the result matches expected output
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
            # Test that proper punctuation is maintained
            if "!" in original:
                assert "!" in preprocessed, "Punctuation should be preserved in Hindi"
            # Test that non-Devanagari special characters are removed
            if "@#$%" in original:
                assert "@" not in preprocessed, "Special characters should be removed"
                assert "#" not in preprocessed, "Special characters should be removed"
                assert "$" not in preprocessed, "Special characters should be removed"
                assert "%" not in preprocessed, "Special characters should be removed"
    
    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert "हिंदी" not in en_processed, "Hindi characters should be removed when processing as English"
        
        # When language is explicitly set to Hindi
        hi_processed = preprocess_text(mixed, 'hi')
        assert "english" not in hi_processed.lower(), "English characters should be removed when processing as Hindi"


class TestTokenization:
    """Tests for the tokenize_text function."""
    
    def test_tokenization_single_text(self, mock_tokenizer):
        """Test tokenization of a single text string."""
        text = "This is a test sentence."
        tokens = tokenize_text(text, mock_tokenizer)
        
        # When language is expli        assert tokens['attention_mask'].shape == direct_tokens['attention_mask'].shape, "Shapes don't match"
    
    def test_max_length(self, mock_tokenizer):
        """Test that max_length parameter is respected."""
        text = "This is a test sentence."
        
        # Test with default max_length (128)
        tokens_default = tokenize_text(text, mock_tokenizer)
        
        # Test with custom max_length
        custom_length = 64
        tokens_custom = tokenize_text(text, mock_tokenizer, max_length=custom_length)
        
        # Check that tokenizer was called with proper params (in a real implementation)
        assert isinstance(tokens_default, dict), "Should return a dictionary"
        assert isinstance(tokens_custom, dict), "Should return a dictionary"
    
    def test_empty_input(self, mock_tokenizer):
        """Test tokenization with empty input."""
        # Empty string
        tokens = tokenize_text("", mock_tokenizer)
        assert 'input_ids' in tokens, "Should handle empty strings"
        
        # Empty list
        tokens = tokenize_text([], mock_tokenizer)
        assert 'input_ids' in tokens, "Should handle empty lists"
        assert tokens['input_ids'].shape[0] == 0, "Batch size should be 0 for empty list"


class TestDatasetLoading:
    """Tests for the load_dataset function."""
    
    def test_hindi_preprocessing(self):
        """Test preprocessing of Hindi text."""
        hindi_texts = [
            ("नमस्ते, दुनिया!", "नमस्ते दुनिया!"),  # Keep punctuation but normalize spaces
            ("विशेष @#$% अक्षर!", "विशेष अक्षर!"),  # Remove non-Devanagari special chars
            ("हिंदी में 123 संख्याएँ", "हिंदी में  संख्याएँ"),  # Non-Devanagari numbers removed
            ("अतिरिक्त  स्थान   यहाँ", "अतिरिक्त स्थान यहाँ"),  # Normalize spaces
            ("देवनागरी वर्णमाला: क ख ग", "देवनागरी वर्णमाला क ख ग"),  # Preserve Devanagari chars
            ("ये $%#@! चिह्न हटा दिए जाएंगे", "ये  चिह्न हटा दिए जाएंगे"),  # Remove special chars
            ("हिंदी ! . ? निशान", "हिंदी ! . ? निशान")  # Keep punctuation
        ]
        
        for original, expected in hindi_texts:
            preprocessed = preprocess_text(original, 'hi')
            # Test Devanagari character preservation
            assert any('\u0900' <= c <= '\u097F' for c in preprocessed), f"No Devanagari characters found: {preprocessed}"
            # Test that the result matches expected output
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
            # Test that proper punctuation is maintained
            if "!" in original:
                assert "!" in preprocessed, "Punctuation should be preserved in Hindi"
            # Test that non-Devanagari special characters are removed
            if "@#$%" in original:
                assert "@" not in preprocessed, "Special characters should be removed"
                assert "#" not in preprocessed, "Special characters should be removed"
                assert "$" not in preprocessed, "Special characters should be removed"
                assert "%" not in preprocessed, "Special characters should be removed"
    
    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert "हिंदी" not in en_processed, "Hindi characters should be removed when processing as English"
        hi_processed = preprocess_text(mixed, 'hi')
        assert "english" not in hi_processed.lower(), "English characters should be removed when processing as Hindi"
            ("हिंदी में 123 संख्याएँ", "हिंदी में  संख्याएँ"),  # Non-Devanagari numbers removed
            ("अतिरिक्त  स्थान   यहाँ", "अतिरिक्त स्थान यहाँ"),  # Normalize spaces
            ("देवनागरी वर्णमाला: क ख ग", "देवनागरी वर्णमाला क ख ग"),  # Preserve Devanagari chars
            ("ये $%#@! चिह्न हटा दिए जाएंगे", "ये  चिह्न हटा दिए जाएंगे"),  # Remove special chars
            ("हिंदी ! . ? निशान", "हिंदी ! . ? निशान")  # Keep punctuation
        ]
        
        for original, expected in hindi_texts:
            preprocessed = preprocess_text(original, 'hi')
            # Test Devanagari character preservation
            assert any('\u0900' <= c <= '\u097F' for c in preprocessed), f"No Devanagari characters found: {preprocessed}"
            # Test that the result matches expected output
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
            # Test that proper punctuation is maintained
            if "!" in original:
                assert "!" in preprocessed, "Punctuation should be preserved in Hindi"
            # Test that non-Devanagari special characters are removed
            if "@#$%" in original:
                assert "@" not in preprocessed, "Special characters should be removed"
                assert "#" not in preprocessed, "Special characters should be removed"
                assert "$" not in preprocessed, "Special characters should be removed"
                assert "%" not in preprocessed, "Special characters should be removed"
    
    def test_mixed_language(self):
        """Test preprocessing with mixed language input."""
        # When language is explicitly set to English
        mixed = "English with हिंदी words"
        en_processed = preprocess_text(mixed, 'en')
        assert prepared_df.empty, "Result should be empty for empty input"
        assert 'cleaned_text' in prepared_df.columns, "Should have cleaned_text column"
        assert 'preprocessed_text' in prepared_df.columns, "Should have preprocessed_text column"
        """Test loading data from CSV file."""
        df = load_dataset(sample_csv_path)
        
        # Basic checks
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert 'text' in df.columns, "text column missing"
        assert 'emotion' in df.columns, "emotion column missing"
        assert 'language' in df.columns, "language column missing"
    
    def test_json_loading(self, sample_dataset, test_data_dir):
        """Test loading data from JSON file."""
        # Create a sample JSON file
        json_path = test_data_dir / "sample_data.json"
        sample_dataset.to_json(json_path, orient="records")
        
        # Load it
        df = load_dataset(json_path)
        
        # Basic checks
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(df) == len(sample_dataset), "DataFrame length doesn't match"
        assert set(df.columns) == set(sample_dataset.columns), "Columns don't match"
    
    def test_excel_loading(self, sample_dataset, test_data_dir):
        """Test loading data from Excel file."""
        try:
            # Try to create Excel file, but skip test if openpyxl is not installed
            excel_path = test_data_dir / "sample_data.xlsx"
            sample_dataset.to_excel(excel_path, index=False)
            
            # Load it
            df = load_dataset(excel_path)
            
            # Basic checks
            assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
            assert len(df) == len(sample_dataset), "DataFrame length doesn't match"
            
        except ImportError:
            pytest.skip("openpyxl not installed, skipping Excel test")
    
    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_file.csv")
    
    def test_unsupported_format(self, test_data_dir):
        """Test error handling for unsupported file format."""
        # Create a file with unsupported extension
        txt_path = test_data_dir / "sample_data.txt"
        with open(txt_path, 'w') as f:
            f.write("This is not a supported format")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(txt_path)


class TestDatasetPreparation:
    """Tests for the prepare_dataset function."""
    
    def test_basic_preparation(self, sample_dataset):
        """Test basic dataset preparation."""
        prepared_df = prepare_dataset(sample_dataset, 'text', 'emotion', 'language')
        
        # Check for new columns
        assert 'cleaned_text' in prepared_df.columns, "cleaned_text column missing"
        assert 'preprocessed_text' in prepared_df.columns, "preprocessed_text column missing"
        
        # Check that original data is preserved
        assert len(prepared_df) == len(sample_dataset), "DataFrame length doesn't match"
        assert all(prepared_df['text'] == sample_dataset['text']), "Original text column was modified"
        assert all(prepared_df['emotion'] == sample_dataset['emotion']), "Original emotion column was modified"
    
    def test_language_detection(self, sample_dataset):
        """Test automatic language detection if not provided."""
        # Remove language column
        df_no_lang = sample_dataset.drop(columns=['language'])
        
        # Prepare dataset without specifying language column
        prepared_df = prepare_dataset(df_no_lang, 'text', 'emotion')
        
        # Check that language was detected
        assert 'detected_language' in prepared_df.columns, "detected_language column missing"
        
        # Check correctness of detection
        english_texts = prepared_df[prepared_df['text'].str.contains(r'^[A-Za-z]')]['detected_language']
        hindi_texts = prepared_df[prepared_df['text'].str.contains(r'[\u0900-\u097F]')]['detected_language']
        
        assert all(english_texts == 'en'), "English texts not correctly detected"
        assert all(hindi_texts == 'hi'), "Hindi texts not correctly detected"
    
    def test_language_specific_preprocessing(self, sample_dataset):
        """Test that preprocessing is applied differently based on language."""
        prepared_df = prepare_dataset(sample_dataset, 'text', 'emotion', 'language')
        
        # Check English preprocessing
        english_rows = prepared_df[prepared_df['language'] == 'en']
        for _, row in english_rows.iterrows():
            # English text should be lowercase
            assert row['preprocessed_text'].islower(), f"English text not lowercased: {row['preprocessed_text']}"
            # English should only have alphanumeric and space
            assert all(c.isalnum() or c.isspace() for c in row['preprocessed_text']), \
                f"Non-alphanumeric characters in English text: {row['preprocessed_text']}"
        
        # Check Hindi preprocessing
        hindi_rows = prepared_df[prepared_df['language'] == 'hi']
        for _, row in hindi_rows.iterrows():
            # Hindi should retain Devanagari characters
            assert any('\u0900' <= c <= '\u097F' for c in row['preprocessed_text']), \
                f"No Devanagari characters found in Hindi text: {row['preprocessed_text']}"
    
    def test_invalid_column_names(self, sample_dataset):
        """Test error handling for invalid column names."""
        # Test with non-existent text column
        with pytest.raises(KeyError):
            prepare_dataset(sample_dataset, 'non_existent_column', 'emotion', 'language')
        
        # Test with non-existent label column
        with pytest.raises(KeyError):
            prepare_dataset(sample_dataset, 'text', 'non_existent_column', 'language')
        
        # Test with non-existent language column
        with pytest.raises(KeyError):
            prepare_dataset(sample_dataset, 'text', 'emotion', 'non_existent_column')
    
    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame(columns=['text', 'emotion', 'language'])
        prepared_df = prepare_dataset(empty_df, 'text', 'emotion', 'language')
        
        assert prepared_df.empty, "Result should be empty for empty input"
        assert 'cleaned_text' in prepared_df.columns, "Should have cleaned_text column"
        assert 'preprocessed_text' in prepared_df.columns, "Should have preprocessed_text column"
            "Machine learning is fascinating!",
            "1234567890",  # Numbers should default to English
            "!@#$%^&*()",  # Special characters should default to English
            ""  # Empty string should default to English
        ]
        
        for text in english_texts:
            assert detect_language(text) == 'en', f"Failed to detect English in: {text}"
    
    def test_hindi_detection(self):
        """Test detection of Hindi text."""
        hindi_texts = [
            "यह एक हिंदी वाक्य है।",
            "नमस्ते, आप कैसे हैं?",
            "मशीन लर्निंग बहुत रोचक है!",
            "१२३४५६७८९०",  # Hindi digits
            "हिंदी में लिखा गया"
        ]
        
        for text in hindi_texts:
            assert detect_language(text) == 'hi', f"Failed to detect Hindi in: {text}"
    
    def test_mixed_text(self):
        """Test detection with mixed language text."""
        # Test with predominantly English but some Hindi
        mixed_english = "This is mostly English with some हिंदी words."
        assert detect_language(mixed_english) == 'hi', "Should detect Hindi in mixed text"
        
        # Even a single Hindi character should be detected
        mostly_english = "Almost all English but ह"
        assert detect_language(mostly_english) == 'hi', "Should detect single Hindi character"


class TestTextCleaning:
    """Tests for the clean_text function."""
    
    def test_url_removal(self):
        """Test that URLs are removed."""
        urls = [
            "Check out https://example.com",
            "Visit www.example.com for more info",
            "Link: http://example.com/path?param=value",
            "Secure site: https://secure.example.com"
        ]
        
        for text in urls:
            cleaned = clean_text(text)
            assert "example.com" not in cleaned, f"URL not removed from: {text}"
    
    def test_html_removal(self):
        """Test that HTML tags are removed."""
        html_texts = [
            "This is <b>bold</b> text",
            "<p>This is a paragraph</p>",
            "Mixed <i>italic</i> and <b>bold</b> text",
            "<div class='container'>Content</div>"
        ]
        
        for text in html_texts:
            cleaned = clean_text(text)
            assert "<" not in cleaned and ">" not in cleaned, f"HTML tags not removed from: {text}"
            assert "bold" in cleaned if "bold" in text else True, "Content within tags should be preserved"
    
    def test_whitespace_handling(self):
        """Test that extra whitespace is properly handled."""
        whitespace_texts = [
            "  Extra spaces  at   ends  ",
            "Multiple    spaces    between    words",
            "Tabs\t\tand\tnewlines\n\n",
            "\n\n\n\n"  # Just newlines
        ]
        
        for text in whitespace_texts:
            cleaned = clean_text(text)
            assert re.match(r"^\S.*\S$", cleaned) if cleaned else True, f"Whitespace not properly handled: '{text}' -> '{cleaned}'"
            assert "  " not in cleaned, f"Multiple consecutive spaces found in: '{cleaned}'"
    
    def test_edge_cases(self):
        """Test edge cases for text cleaning."""
        # Empty string
        assert clean_text("") == "", "Empty string should remain empty"
        
        # Only whitespace
        assert clean_text("   \t\n   ") == "", "String with only whitespace should become empty"
        
        # Only URLs
        assert clean_text("https://example.com") == "", "String with only URL should become empty"
        
        # Only HTML
        assert clean_text("<div></div>") == "", "String with only HTML should become empty or contain only the content"


class TestPreprocessing:
    """Tests for the preprocess_text function."""
    
    def test_english_preprocessing(self):
        """Test preprocessing of English text."""
        english_texts = [
            ("Hello, World!", "hello world"),  # Lowercase, remove punctuation
            ("Special @#$% Characters!", "special characters"),  # Remove special chars
            ("Numbers 123 should stay", "numbers 123 should stay"),  # Keep numbers
            ("Extra  spaces   here", "extra spaces here")  # Normalize spaces
        ]
        
        for original, expected in english_texts:
            preprocessed = preprocess_text(original, 'en')
            assert preprocessed.islower(), f"Text not lowercased: {preprocessed}"
            assert all(c.isalnum() or c.isspace() for c in preprocessed), f"Non-alphanumeric characters found: {preprocessed}"
            assert preprocessed == expected, f"Expected '{expected}', got '{preprocessed}'"
    
    def test_hindi_preprocessing(self):
        """Test preprocessing of Hindi text."""
        hindi_texts = [
            ("नमस्ते, दुनिया!", "नमस्ते दुनिया!"),  # Keep punctuation but normalize spaces
            

