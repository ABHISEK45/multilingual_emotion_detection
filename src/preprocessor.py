"""
Text preprocessing module for multilingual emotion detection.

This module handles text preprocessing for English and Hindi languages,
including text cleaning, tokenization, and language detection.
"""

import re
import logging
import unicodedata
import pandas as pd
from typing import Optional, Dict, List, Union, Any, Tuple
from pathlib import Path

# Import language detection libraries
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = {'en', 'hi'}

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Uses langdetect library if available, otherwise falls back to a simple heuristic
    based on Unicode character ranges.

    Args:
        text (str): Input text

    Returns:
        str: Detected language code ('en' for English, 'hi' for Hindi)
        
    Raises:
        ValueError: If text is empty or None
    """
    # Input validation
    if not text or not isinstance(text, str):
        logger.error("Invalid input: Text must be a non-empty string")
        raise ValueError("Text must be a non-empty string")
    
    # Remove whitespace for better detection
    text = text.strip()
    if not text:
        logger.error("Invalid input: Text contains only whitespace")
        raise ValueError("Text contains only whitespace")
    
    # Try using langdetect if available
    if LANGDETECT_AVAILABLE:
        try:
            # Attempt to detect language with langdetect
            detected = detect(text)
            logger.debug(f"langdetect detected language: {detected}")
            
            # Map langdetect codes to our supported codes
            if detected == 'hi':
                return 'hi'
            elif detected.startswith('en'):
                return 'en'
        except LangDetectException as e:
            logger.warning(f"langdetect failed: {str(e)}. Falling back to heuristic method.")
    
    # Fallback to heuristic approach
    # Check for Devanagari Unicode range for Hindi
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_chars = len(re.findall(devanagari_pattern, text))
    
    # If more than 25% of characters are Devanagari, consider it Hindi
    if devanagari_chars > len(text.replace(" ", "")) * 0.25:
        logger.debug(f"Heuristic detected Hindi: {devanagari_chars} Devanagari characters found")
        return 'hi'
    
    # Default to English
    logger.debug("Language defaulted to English")
    return 'en'


def validate_text(text: str) -> Tuple[bool, str]:
    """
    Validate if the input text is suitable for processing.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_valid, error_message) where is_valid is a boolean and
               error_message is a string describing the error (if any)
    """
    if not isinstance(text, str):
        return False, "Input must be a string"
    
    if not text:
        return False, "Input text is empty"
    
    if text.strip() == "":
        return False, "Input text contains only whitespace"
    
    # Check for minimum meaningful content (at least 2 characters excluding spaces)
    if len(text.strip()) < 2:
        return False, "Input text is too short (minimum 2 characters required)"
        
    return True, ""


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra spaces, etc.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
        
    Raises:
        ValueError: If text is empty, None or invalid
    """
    # Validate input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for clean_text: {error_message}")
        raise ValueError(f"Invalid input: {error_message}")
    
    logger.debug(f"Cleaning text: {text[:50]}...")
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters that aren't relevant for emotion detection
    text = re.sub(r'[^\w\s\u0900-\u097F\.\,\!\?\-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    logger.debug(f"Cleaned text: {text[:50]}...")
    return text


def detect_mixed_script_hindi(text: str) -> bool:
    """
    Detect if text contains Hindi content written in Latin script.
    
    This is useful for identifying Hinglish text or romanized Hindi.
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if the text appears to be Hindi written in Latin script
    """
    # Common Hindi words and expressions written in Latin script
    hindi_latin_markers = [
        "namaste", "namaskar", "dhanyavad", "shukriya", 
        "kaise", "kaisa", "kaisi", "kya", "kyun", "kyon",
        "aap", "tum", "acha", "accha", "thik", "theek", 
        "hai", "hain", "nahi", "nahin", "karo", "karna",
        "chahiye", "bahut", "bohot", "jyada"
    ]
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Check for common Hindi words in Latin script
    for marker in hindi_latin_markers:
        if re.search(r'\b' + marker + r'\b', text_lower):
            return True
    
    # Check for common Hindi language patterns
    hindi_patterns = [
        r'\b(kya|kyun|kaise)\b.*\b(hai|hain|tha|thi|the)\b',  # Question patterns
        r'\b(main|mein|hum|aap|tum).*\b(hai|hain|tha|thi|the)\b',  # Subject-verb patterns
        r'\b(ka|ki|ke)\b',  # Possessive markers
    ]
    
    for pattern in hindi_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def validate_hindi_text(text: str) -> Tuple[bool, str]:
    """
    Validate if the text contains valid Hindi characters.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and
               message contains validation details
    """
    # Check if text contains any Devanagari characters
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if not devanagari_pattern.search(text):
        # If no Devanagari characters, check if it might be romanized Hindi
        if detect_mixed_script_hindi(text):
            return True, "Text appears to be Hindi written in Latin script"
        return False, "No Hindi characters found in text"
    
    # Calculate percentage of Devanagari characters in text
    total_chars = len(text.replace(" ", ""))
    devanagari_chars = len(re.findall(devanagari_pattern, text))
    devanagari_percentage = (devanagari_chars / total_chars) * 100 if total_chars > 0 else 0
    
    if devanagari_percentage < 50:
        return False, f"Text contains only {devanagari_percentage:.1f}% Hindi characters"
    
    return True, f"Valid Hindi text with {devanagari_percentage:.1f}% Devanagari characters"


def normalize_hindi_text(text: str) -> str:
    """
    Normalize Hindi text for better processing.
    
    Performs Unicode normalization and handles character variants.
    
    Args:
        text (str): Hindi input text
        
    Returns:
        str: Normalized Hindi text
    """
    # Check for mixed script Hindi (romanized Hindi)
    if detect_mixed_script_hindi(text) and not re.search(r'[\u0900-\u097F]', text):
        logger.info("Detected romanized Hindi text. Processing as is.")
        # For romanized Hindi, we keep the text as is but normalize spaces
        return re.sub(r'\s+', ' ', text).strip()
    
    # Apply Unicode normalization (NFD followed by NFC)
    text = unicodedata.normalize('NFC', unicodedata.normalize('NFD', text))
    
    # More comprehensive character normalization mappings for Hindi
    char_maps = {
        # Variant characters
        '\u0931': '\u090B',  # Variant of ऋ
        '\u0904': '\u0901',  # Variant of ँ (chandrabindu)
        '\u0950': '',        # Remove ॐ symbol
        
        # Punctuation normalization
        '\u0964': '.',       # Replace Danda with period
        '\u0965': '...',     # Replace Double Danda with ellipsis
        
        # Nukta normalization
        '\u0929': '\u0928',  # ऩ → न
        '\u0931': '\u0930',  # ऱ → र
        '\u0934': '\u0933',  # ऴ → ळ
        
        # Common combining characters normalization
        '\u093C': '',        # ़ (nukta) - only normalize standalone nukta
    }
    
    # Apply character mappings
    for char, replacement in char_maps.items():
        text = text.replace(char, replacement)
    
    # Handle common variations of vowel combinations
    vowel_variations = {
        'े\u093E': 'े',  # े + ा → े
        'े\u0947': 'े',  # े + े → े
        'ो\u093E': 'ो',  # ो + ा → ो
    }
    for var, repl in vowel_variations.items():
        text = text.replace(var, repl)
    
    # Remove ZWJ and ZWNJ characters
    text = text.replace('\u200D', '').replace('\u200C', '')
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s*([।,.!?])\s*', r'\1 ', text)
    text = text.strip()
    
    return text


def preprocess_text(text: str, language: Optional[str] = None) -> str:
    """
    Preprocess text based on language.

    Args:
        text (str): Input text
        language (str, optional): Language code ('en' for English, 'hi' for Hindi).
                                If None, language will be auto-detected.

    Returns:
        str: Preprocessed text ready for model input
        
    Raises:
        ValueError: If text is empty/invalid or language is unsupported
        
    Example:
        >>> preprocess_text("Hello, how are you?", "en")
        'hello how are you'
        >>> preprocess_text("नमस्ते, आप कैसे हैं?", "hi")
        'नमस्ते आप कैसे हैं'
    """
    # Validate input text
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for preprocess_text: {error_message}")
        raise ValueError(f"Invalid input: {error_message}")
    
    # Auto-detect language if not provided
    if language is None:
        try:
            language = detect_language(text)
            logger.info(f"Auto-detected language: {language}")
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise ValueError(f"Language detection failed: {str(e)}")
    
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        logger.error(f"Unsupported language: {language}")
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}")
    
    logger.debug(f"Preprocessing text in {language}: {text[:50]}...")
    
    # Clean the text
    # Clean the text
    text = clean_text(text)
    
    if language == 'en':
        # English-specific preprocessing
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters (but keep spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Standardize contractions (optional enhancement)
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
    
    elif language == 'hi':
        # Hindi-specific preprocessing
        
        # Check for romanized Hindi or mixed script text
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        devanagari_chars = len(re.findall(devanagari_pattern, text))
        non_space_chars = len(text.replace(" ", ""))
        
        # If text has very few or no Devanagari characters, check if it's romanized Hindi
        if devanagari_chars < non_space_chars * 0.25 and detect_mixed_script_hindi(text):
            logger.info("Detected potential romanized Hindi text. Attempting conversion.")
            # Try to convert romanized Hindi to Devanagari
            text = process_hinglish_text(text)
        # If text has both Devanagari and Latin script, handle as mixe
        # Normalize Hindi numerals to standard form
        hindi_digit_map = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        for hindi_digit, latin_digit in hindi_digit_map.items():
            text = text.replace(hindi_digit, latin_digit)
        
        # Handle common Hindi punctuation marks
        text = text.replace('।', '.').replace('॥', '.')
    
    # Final cleaning for both languages
    text = re.sub(r'\s+', ' ', text).strip()
    
    logger.debug(f"Preprocessed text: {text[:50]}...")
    return text

def tokenize_text(text: str, tokenizer, max_length: int = 128):
    """
    Tokenize text using the provided tokenizer.

    Args:
        text (str): Input text
        tokenizer: The tokenizer object from transformers
        max_length (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        dict: Tokenized inputs ready for the model
        
    Raises:
        ValueError: If text is invalid or tokenizer fails
    """
    # Validate input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for tokenize_text: {error_message}")
        raise ValueError(f"Invalid input: {error_message}")
    
    try:
        # Use the provided tokenizer
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        logger.debug(f"Tokenized text with {len(tokens['input_ids'][0])} tokens")
        return tokens
        
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise ValueError(f"Tokenization failed: {str(e)}")


def load_dataset(dataset_path):
    """
    Load dataset from a file.

    Args:
        dataset_path (str): Path to the dataset file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix == '.json':
        df = pd.read_json(path)
    elif path.suffix == '.xlsx' or path.suffix == '.xls':
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return df


def prepare_dataset(df, text_column, label_column=None, language_column=None):
    """
    Prepare dataset for training or evaluation.

    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the column containing text
        label_column (str, optional): Name of the column containing emotion labels.
                                     Defaults to None.
        language_column (str, optional): Name of the column containing language codes.
                                        Defaults to None.

    Returns:
        pd.DataFrame: Prepared dataset
    """
    # Create a copy to avoid modifying the original
    prepared_df = df.copy()
    
    # Clean text
    prepared_df['cleaned_text'] = prepared_df[text_column].apply(clean_text)
    
    # Detect language if not provided
    if language_column is None:
        prepared_df['detected_language'] = prepared_df[text_column].apply(detect_language)
        language_column = 'detected_language'
    
    # Preprocess text based on language
    prepared_df['preprocessed_text'] = prepared_df.apply(
        lambda row: preprocess_text(row['cleaned_text'], row[language_column]), 
        axis=1
    )
    
    # Add Hindi validation for Hindi texts
    if 'hi' in prepared_df[language_column].values:
        # Only validate texts identified as Hindi
        hindi_mask = prepared_df[language_column] == 'hi'
        hindi_validation = prepared_df.loc[hindi_mask, 'cleaned_text'].apply(validate_hindi_text)
        
        # Extract validation results
        prepared_df.loc[hindi_mask, 'hindi_valid'] = hindi_validation.apply(lambda x: x[0])
        prepared_df.loc[hindi_mask, 'hindi_validation_message'] = hindi_validation.apply(lambda x: x[1])
        
        # Log validation results
        invalid_hindi = prepared_df[hindi_mask & ~prepared_df['hindi_valid']]
        if len(invalid_hindi) > 0:
            logger.warning(f"Found {len(invalid_hindi)} texts with potentially invalid Hindi content")
    
    return prepared_df


def transliterate_to_devanagari(text: str) -> str:
    """
    Transliterate Hindi text written in Latin script (romanized Hindi) to Devanagari script.
    
    This function implements a rule-based transliteration system to convert romanized
    Hindi (also known as Hinglish or Latin-script Hindi) to Devanagari script. It follows
    common transliteration conventions while handling special cases like conjunct 
    characters and vowel modifiers.
    
    Args:
        text (str): Hindi text written in Latin script
        
    Returns:
        str: Transliterated text in Devanagari script
        
    Raises:
        ValueError: If the input text is empty or invalid
        
    Examples:
        >>> transliterate_to_devanagari("namaste")
        'नमस्ते'
        >>> transliterate_to_devanagari("main aapse baat karna chahta hoon")
        'मैं आपसे बात करना चाहता हूँ'
        >>> transliterate_to_devanagari("pyaar")
        'प्यार'
    """
    # Validate input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for transliteration: {error_message}")
        raise ValueError(f"Invalid input: {error_message}")
        
    # Check if text already contains Devanagari characters (more than 25%)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_chars = len(re.findall(devanagari_pattern, text))
    non_space_chars = len(text.replace(" ", ""))
    
    if devanagari_chars > 0 and (devanagari_chars / non_space_chars) > 0.25:
        logger.info("Text already contains significant Devanagari characters. Skipping transliteration.")
        return text
    
    # Pre-process the text
    text = text.lower()
    
    # Define conversion mappings for vowels
    vowels = {
        'a': '\u0905',    # अ
        'aa': '\u0906',   # आ
        'i': '\u0907',    # इ
        'ee': '\u0908',   # ई
        'u': '\u0909',    # उ
        'oo': '\u090A',   # ऊ
        'ri': '\u090B',   # ऋ
        'e': '\u090F',    # ए
        'ai': '\u0910',   # ऐ
        'o': '\u0913',    # ओ
        'au': '\u0914',   # औ
    }
    
    # Define vowel modifiers (matras)
    vowel_modifiers = {
        'a': '',           # No modifier for 'a'
        'aa': '\u093E',    # ा
        'i': '\u093F',     # ि
        'ee': '\u0940',    # ी
        'u': '\u0941',     # ु
        'oo': '\u0942',    # ू
        'ri': '\u0943',    # ृ
        'e': '\u0947',     # े
        'ai': '\u0948',    # ै
        'o': '\u094B',     # ो
        'au': '\u094C',    # ौ
    }
    
    # Define consonants
    consonants = {
        'k': '\u0915',    # क
        'kh': '\u0916',   # ख
        'g': '\u0917',    # ग
        'gh': '\u0918',   # घ
        'ng': '\u0919',   # ङ
        'ch': '\u091A',   # च
        'chh': '\u091B',  # छ
        'j': '\u091C',    # ज
        'jh': '\u091D',   # झ
        'n': '\u0928',    # न (default n)
        'ny': '\u091E',   # ञ
        't': '\u0924',    # त (dental)
        'th': '\u0925',   # थ (dental)
        'd': '\u0926',    # द (dental)
        'dh': '\u0927',   # ध (dental)
        'tt': '\u091F',   # ट (retroflex)
        'tth': '\u0920',  # ठ (retroflex)
        'dd': '\u0921',   # ड (retroflex)
        'ddh': '\u0922',  # ढ (retroflex)
        'nn': '\u0923',   # ण (retroflex)
        'p': '\u092A',    # प
        'ph': '\u092B',   # फ
        'f': '\u092B',    # फ (alternative for ph)
        'b': '\u092C',    # ब
        'bh': '\u092D',   # भ
        'm': '\u092E',    # म
        'y': '\u092F',    # य
        'r': '\u0930',    # र
        'l': '\u0932',    # ल
        'v': '\u0935',    # व
        'w': '\u0935',    # व (alternative for v)
        'sh': '\u0936',   # श
        'ss': '\u0937',   # ष
        's': '\u0938',    # स
        'h': '\u0939',    # ह
    }
    
    # Define special characters
    special_chars = {
        '.': '।',         # Danda (period)
        ',': ',',         # Comma
        '?': '?',         # Question mark
        '!': '!',         # Exclamation mark
        ';': ';',         # Semicolon
        ':': ':',         # Colon
        '"': '"',         # Quotation mark
        "'": "'",         # Apostrophe
        '-': '-',         # Hyphen
        '(': '(',         # Left parenthesis
        ')': ')',         # Right parenthesis
    }
    
    # Define special conjunct mappings
    special_conjuncts = {
        'ksh': '\u0915\u094D\u0937',  # क्ष
        'tr': '\u0924\u094D\u0930',   # त्र
        'gya': '\u091C\u094D\u091E',  # ज्ञ
        'dny': '\u091C\u094D\u091E',  # ज्ञ (alternative)
        'shr': '\u0936\u094D\u0930',  # श्र
    }
    
    # Define special word mappings
    special_words = {
        'om': '\u0913\u092E\u094D',  # ॐ
        'shri': '\u0936\u094D\u0930\u0940',  # श्री
    }
    
    # Helper function to check if a substring matches at a position
    def matches_at(substr, pos):
        return text[pos:pos+len(substr)] == substr
    
    # Process the text
    result = []
    i = 0
    prev_was_consonant = False
    
    try:
        # First check for special words
        for word in special_words:
            text = re.sub(r'\b' + word + r'\b', f" {special_words[word]} ", text)
        
        text = text.strip()
        
        while i < len(text):
            # Skip spaces
            if text[i] == ' ':
                result.append(' ')
                prev_was_consonant = False
                i += 1
                continue
                
            # Check for special characters
            if text[i] in special_chars:
                result.append(special_chars[text[i]])
                prev_was_consonant = False
                i += 1
                continue
                
            # Check for numerals (keep as is)
            if text[i].isdigit():
                result.append(text[i])
                prev_was_consonant = False
                i += 1
                continue
                
            # Check for special conjuncts
            found_conjunct = False
            for conjunct in sorted(special_conjuncts.keys(), key=len, reverse=True):
                if i + len(conjunct) <= len(text) and matches_at(conjunct, i):
                    result.append(special_conjuncts[conjunct])
                    i += len(conjunct)
                    prev_was_consonant = True
                    found_conjunct = True
                    break
                    
            if found_conjunct:
                continue
                
            # Check for consonants (including digraphs like 'kh', 'gh', etc.)
            found_consonant = False
            for consonant in sorted(consonants.keys(), key=len, reverse=True):
                if i + len(consonant) <= len(text) and matches_at(consonant, i):
                    # Check if followed by a vowel modifier
                    next_pos = i + len(consonant)
                    has_vowel_modifier = False
                    
                    for vowel in sorted(vowel_modifiers.keys(), key=len, reverse=True):
                        if next_pos + len(vowel) <= len(text) and matches_at(vowel, next_pos):
                            # Add consonant with vowel modifier
                            if vowel == 'a':  # Skip 'a' vowel as it's implicit
                                result.append(consonants[consonant])
                            else:
                                result.append(consonants[consonant] + vowel_modifiers[vowel])
                            i = next_pos + len(vowel)
                            has_vowel_modifier = True
                            break
                            
                    if not has_vowel_modifier:
                        # No vowel follows - add halant unless followed by another consonant
                        if next_pos < len(text) and text[next_pos] != ' ' and any(matches_at(c, next_pos) for c in consonants):
                            result.append(consonants[consonant] + '\u094D')  # Add halant
                        else:
                            result.append(consonants[consonant])
                        i = next_pos
                        
                    prev_was_consonant = True
                    found_consonant = True
                    break
                    
            if found_consonant:
                continue
                
            # Check for vowels
            found_vowel = False
            for vowel in sorted(vowels.keys(), key=len, reverse=True):
                if i + len(vowel) <= len(text) and matches_at(vowel, i):
                    # If previous character was a consonant, use vowel modifier
                    if prev_was_consonant and vowel != 'a':
                        result.append(vowel_modifiers[vowel])
                    else:
                        result.append(vowels[vowel])
                    i += len(vowel)
                    prev_was_consonant = False
                    found_vowel = True
                    break
                    
            if found_vowel:
                continue
                
            # If no match was found, keep the character as is
            result.append(text[i])
            prev_was_consonant = False
            i += 1
            
        transliterated_text = ''.join(result)
        logger.info(f"Transliterated text from Latin to Devanagari script")
        return transliterated_text
        
    except Exception as e:
        logger.error(f"Transliteration failed: {str(e)}")
        # Return original text if transliteration fails
        return text


def detect_and_convert_romanized_hindi(text: str) -> str:
    """
    Detect if the text is romanized Hindi and convert it to Devanagari if needed.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text converted to Devanagari if it was romanized Hindi,
             otherwise returns the original text
    """
    try:
        # First check if text is already primarily in Devanagari
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        devanagari_chars = len(re.findall(devanagari_pattern, text))
        non_space_chars = len(text.replace(" ", ""))
        
        # If more than 25% is already Devanagari, leave it as is
        if devanagari_chars > 0 and (devanagari_chars / non_space_chars) > 0.25:
            return text
            
        # Check if it might be romanized Hindi
        if detect_mixed_script_hindi(text):
            logger.info("Detected text as potential romanized Hindi. Attempting transliteration.")
            transliterated = transliterate_to_devanagari(text)
            
            # Validate the transliteration result
            valid, message = validate_hindi_text(transliterated)
            if valid:
                logger.info(f"Transliteration successful: {message}")
                return transliterated
            else:
                logger.warning(f"Transliteration produced invalid Hindi text: {message}. Using original text.")
                return text
                
        return text
        
    except Exception as e:
        logger.error(f"Error during romanized Hindi detection and conversion: {str(e)}")
        return text


def handle_mixed_script_text(text: str) -> str:
    """
    Handle text containing a mixture of Devanagari and Latin scripts.
    
    This function detects mixed script portions and processes them appropriately,
    transliterating Latin script portions that appear to be Hindi.
    
    Args:
        text (str): Input text with potentially mixed scripts
        
    Returns:
        str: Processed text with consistent script usage
        
    Examples:
        >>> handle_mixed_script_text("नमस्ते, how are you?")
        'नमस्ते, how are you?'
        >>> handle_mixed_script_text("मैंने apna kaam finish कर लिया है")
        'मैंने अपना काम फिनिश कर लिया है'
    """
    # Validate input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for mixed script handling: {error_message}")
        return text
    
    try:
        # Split text into words
        words = text.split()
        result = []
        
        # Process each word
        for word in words:
            # Count Devanagari characters in the word
            devanagari_pattern = re.compile(r'[\u0900-\u097F]')
            devanagari_chars = len(re.findall(devanagari_pattern, word))
            
            # Skip words that are already primarily Devanagari
            if devanagari_chars > 0 and devanagari_chars / len(word) > 0.5:
                result.append(word)
                continue
                
            # Skip English words and non-Hindi terms
            english_pattern = re.compile(r'^[a-zA-Z]+$')
            if english_pattern.match(word) and not is_likely_romanized_hindi(word):
                result.append(word)
                continue
                
            # Try to transliterate potential Hindi words
            if is_likely_romanized_hindi(word):
                transliterated = transliterate_to_devanagari(word)
                result.append(transliterated)
            else:
                result.append(word)
                
        return ' '.join(result)
        
    except Exception as e:
        logger.error(f"Mixed script handling failed: {str(e)}")
        return text


def is_likely_romanized_hindi(word: str) -> bool:
    """
    Determine if a word is likely to be romanized Hindi rather than English.
    
    Args:
        word (str): A single word to analyze
        
    Returns:
        bool: True if the word is likely romanized Hindi
    """
    word = word.lower()
    
    # Common Hindi word endings
    hindi_endings = ['aa', 'ee', 'oo', 'ai', 'au', 'an', 'en', 'on', 
                     'ah', 'ey', 'ki', 'ka', 'ke', 'na', 'ne', 'ni', 
                     'ta', 'te', 'ti', 'ya', 'ye', 'yi', 'ga', 'gi', 'ge']
    
    # Specific Hindi words that might be confused with English
    hindi_specific_words = {
        'main', 'mein', 'hum', 'tum', 'aap', 'yeh', 'woh', 'kya', 
        'kyun', 'kaise', 'kaun', 'kab', 'kahan', 'jab', 'tab', 'hai', 
        'hain', 'tha', 'the', 'thi', 'kar', 'karo', 'karna', 'karna',
        'baat', 'log', 'kuch', 'acha', 'accha', 'thik', 'theek'
    }
    
    # Hindi character patterns
    hindi_patterns = [
        r'[aeiou]{2,}',  # Double vowels common in Hindi transliteration
        r'[bcdfghjklmnpqrstvwxyz]{2,}[aeiou]',  # Consonant clusters followed by vowel
    ]
    
    # Check for exact Hindi words
    if word in hindi_specific_words:
        return True
        
    # Check for Hindi word endings
    for ending in hindi_endings:
        if word.endswith(ending) and len(word) > len(ending):
            return True
            
    # Check for Hindi character patterns
    for pattern in hindi_patterns:
        if re.search(pattern, word):
            # Further verify it's not a common English word
            common_english = {'please', 'thank', 'hello', 'good', 'bad', 'nice', 
                             'great', 'poor', 'happy', 'sad', 'angry', 'fear', 
                             'love', 'hate', 'feel', 'need', 'want', 'give', 
                             'take', 'come', 'going', 'see', 'look', 'watch'}
            if word not in common_english:
                return True
    
    return False


def process_hinglish_text(text: str) -> str:
    """
    Process Hinglish text (a mix of Hindi and English).
    
    This function intelligently handles Hinglish text, preserving English portions
    while transliterating Hindi portions written in Latin script.
    
    Args:
        text (str): Hinglish text input
        
    Returns:
        str: Processed text with Hindi portions in Devanagari
        
    Examples:
        >>> process_hinglish_text("Main office mein busy hoon")
        'मैं office में busy हूँ'
    """
    # Validate input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        logger.error(f"Invalid input for Hinglish processing: {error_message}")
        return text
        
    try:
        # Process common Hinglish patterns
        hinglish_patterns = {
            # Common phrases
            r'\b(kya|kyun|kaise) (hai|hain|tha|thi|the)\b': True,  # Questions
            r'\b(mujhe|maine|mera|meri|mere) ([a-z]+)\b': True,    # First person
            r'\b(aapka|aapke|aapki|tumhara|tumhari|tumhare) ([a-z]+)\b': True,  # Second person
            r'\b([a-z]+) (karna|karke|karunga|karungi|karega|karegi)\b': True,  # Verb forms
            
            # Words to exclude from transliteration
            r'\b(ok|yes|no|please|thank|sorry|hello|hi|bye)\b': False,  # Common English
            r'\b(email|phone|website|internet|computer|file)\b': False,  # Tech terms
            r'\b(meeting|office|manager|team|project|work)\b': False,    # Work terms
        }
        
        # First handle the patterns
        marked_for_transliteration = []
        words = text.split()
        
        # Mark words for transliteration based on patterns
        for i, word in enumerate(words):
            should_transliterate = is_likely_romanized_hindi(word)
            
            # Check for pattern matches
            for pattern, pattern_value in hinglish_patterns.items():
                if re.search(pattern, ' '.join(words[max(0, i-2):min(len(words), i+3)])):
                    should_transliterate = pattern_value
                    break
                    
            marked_for_transliteration.append(should_transliterate)
            
        # Process the text based on the marked words
        result = []
        for i, (word, should_transliterate) in enumerate(zip(words, marked_for_transliteration)):
            if should_transliterate:
                # Skip words with numbers or special characters
                if re.search(r'[0-9@#$%^&*()_+=\[\]{}|\\:;<>,.?/~`]', word):
                    result.append(word)
                    continue
                
                transliterated = transliterate_to_devanagari(word)
                result.append(transliterated)
            else:
                result.append(word)
                
        processed_text = ' '.join(result)
        
        # Validate the final result
        _, message = validate_hindi_text(processed_text)
        logger.info(f"Hinglish processing complete: {message}")
        
        return processed_text
        
    except Exception as e:
        logger.error(f"Hinglish processing failed: {str(e)}")
        return text
