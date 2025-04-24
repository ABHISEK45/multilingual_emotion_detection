"""
Model module for multilingual emotion detection.

This module handles loading, inference, and saving of transformer-based
models for emotion detection in multiple languages.
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Define emotion classes
EMOTION_CLASSES = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]


class EmotionDetectionModel:
    """
    Emotion detection model using transformer-based architecture.
    
    Handles loading pretrained models and performing inference for
    emotion detection in English and Hindi text.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the emotion detection model.

        Args:
            model_path (str, optional): Path to load the model from. 
                                      If None, will load a default pretrained model.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                                  If None, will automatically detect.
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default model if none provided
        if model_path is None:
            # For multilingual emotion detection, XLM-RoBERTa is a good choice
            self.model_name = "xlm-roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # In a real implementation, you'd fine-tune this model on emotion data
            # Here we're just loading the base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=len(EMOTION_CLASSES)
            )
        else:
            # Load model from path
            model_path = Path(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, text):
        """
        Predict emotions for a single text input.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Dictionary containing emotion probabilities
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()[0]
        
        # Create result dictionary
        result = {
            'text': text,
            'emotions': {emotion: float(prob) for emotion, prob in zip(EMOTION_CLASSES, probs)},
            'dominant_emotion': EMOTION_CLASSES[np.argmax(probs)]
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict emotions for a batch of texts.

        Args:
            texts (list): List of text strings to analyze

        Returns:
            list: List of dictionaries containing emotion probabilities for each text
        """
        # Tokenize all texts
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()
        
        # Create result dictionaries
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'emotions': {emotion: float(prob) for emotion, prob in zip(EMOTION_CLASSES, probs[i])},
                'dominant_emotion': EMOTION_CLASSES[np.argmax(probs[i])]
            }
            results.append(result)
        
        return results
    
    def save_model(self, save_path):
        """
        Save the model to disk.

        Args:
            save_path (str): Path to save the model to
        """
        # Create directory if it doesn't exist
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer

