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
            

