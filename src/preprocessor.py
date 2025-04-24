"""
Text preprocessing module for multilingual emotion detection.

This module handles text preprocessing for English and Hindi languages,
including text cleaning, tokenization, and language detection.
"""

import re
import pandas as pd
from pathlib import Path


def detect_language(text):
    """
    Detect the language of the given text.

    Args:
        text (str): Input text

    Returns:
        str: Detected language code ('en' for English, 'hi' for Hindi)
    """
    # This is a simple heuristic approach. In a production system,
    # consider using a dedicated language detection library like langdetect
    
    # Check for Devanagari Unicode range for Hindi
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(text):
        return 'hi'
    
    # Default to English
    return 'en'


def clean_text(text):
    """
    Clean text by removing special characters, extra spaces, etc.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    # Convert to lowercase for English
    # Note: For Hindi, case is not applicable
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(text, language):
    """
    Preprocess text based on language.

    Args:
        text (str): Input text
        language (str): Language code ('en' for English, 'hi' for Hindi)

    Returns:
        str: Preprocessed text ready for model input
    """
    # Clean the text
    text = clean_text(text)
    
    if language == 'en':
        # English-specific preprocessing
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters (but keep spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
    
    elif language == 'hi':
        # Hindi-specific preprocessing
        # Remove non-Devanagari characters (except spaces and punctuation)
        text = re.sub(r'[^\u0900-\u097F\s\.\,\!\?\-]', '', text)
    
    return text


def tokenize_text(text, tokenizer, max_length=128):
    """
    Tokenize text using the provided tokenizer.

    Args:
        text (str): Input text
        tokenizer: The tokenizer object from transformers
        max_length (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        dict: Tokenized inputs ready for the model
    """
    # This is a placeholder. In an actual implementation, you would use
    # a tokenizer from the transformers library
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


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
    
    return prepared_df

