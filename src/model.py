"""
Model module for multilingual emotion detection.

This module handles loading, inference, and saving of transformer-based
models for emotion detection in multiple languages.
"""

import os
import json
import torch
import numpy as np
import logging
import time
import gc
import psutil
import tqdm
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union, Any, Tuple, Callable
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Define emotion classes
EMOTION_CLASSES = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model version
MODEL_VERSION = "1.0.0"


class EmotionDetectionModel:
    """
    Emotion detection model using transformer-based architecture.
    
    Handles loading pretrained models and performing inference for
    emotion detection in English and Hindi text.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the emotion detection model.

        Args:
            model_path (str, optional): Path to load the model from. 
                                      If None, will load a default pretrained model.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                                  If None, will automatically detect.
                                  
        Raises:
            OSError: If the model_path exists but the model files are corrupted or incompatible
            ValueError: If the model configuration is invalid
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model metadata
        self.metadata = {
            "version": MODEL_VERSION,
            "creation_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Default model if none provided
        if model_path is None:
            logger.info("No model path provided, loading default model")
            try:
                # For multilingual emotion detection, XLM-RoBERTa is a good choice
                self.model_name = "xlm-roberta-base"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # In a real implementation, you'd fine-tune this model on emotion data
                # Here we're just loading the base model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=len(EMOTION_CLASSES)
                )
                logger.info(f"Successfully loaded default model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load default model: {str(e)}")
                raise RuntimeError(f"Failed to load default model: {str(e)}")
        else:
            logger.info(f"Loading model from path: {model_path}")
            try:
                # Load model from path
                model_path = Path(model_path)
                
                # Check if metadata file exists and load it
                metadata_path = model_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded model metadata. Version: {self.metadata.get('version', 'unknown')}")
                
                # Load model and tokenizer
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Successfully loaded model from: {model_path}")
            except (OSError, ValueError) as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                
                # Try to load a fallback model if custom model fails
                logger.info("Attempting to load default model as fallback")
                try:
                    self.model_name = "xlm-roberta-base"
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name, 
                        num_labels=len(EMOTION_CLASSES)
                    )
                    logger.warning(f"Loaded default model as fallback after failure to load from {model_path}")
                except Exception as fallback_error:
                    logger.error(f"Fallback model loading also failed: {str(fallback_error)}")
                    raise RuntimeError(f"Failed to load model from {model_path} and fallback model also failed")
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict emotions for a single text input.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Dictionary containing emotion probabilities
            
        Raises:
            ValueError: If input text is empty or None
            RuntimeError: If prediction fails due to model issues
        """
        # Validate input
        if not text or not isinstance(text, str):
            logger.error("Invalid input: Text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")
        
        try:
            logger.debug(f"Predicting emotions for text: {text[:50]}...")
            
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
                'dominant_emotion': EMOTION_CLASSES[np.argmax(probs)],
                'model_version': self.metadata.get('version', MODEL_VERSION)
            }
            
            logger.debug(f"Prediction result: dominant emotion = {result['dominant_emotion']}")
            return result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            dict: Memory usage information with the following keys:
                 - system_used_gb: System memory used in GB
                 - system_available_gb: System memory available in GB
                 - system_percent: System memory usage percentage
                 - cuda_used_gb: CUDA memory used in GB (if available)
                 - cuda_available_gb: CUDA memory available in GB (if available)
        """
        memory_stats = {}
        
        # Get system memory stats
        try:
            memory = psutil.virtual_memory()
            memory_stats["system_used_gb"] = memory.used / (1024 ** 3)
            memory_stats["system_available_gb"] = memory.available / (1024 ** 3)
            memory_stats["system_percent"] = memory.percent
        except Exception as e:
            logger.warning(f"Could not get system memory stats: {str(e)}")
            memory_stats["system_used_gb"] = 0
            memory_stats["system_available_gb"] = 0
            memory_stats["system_percent"] = 0
        
        # Get CUDA memory stats if available
        if torch.cuda.is_available():
            try:
                cuda_id = torch.cuda.current_device()
                memory_stats["cuda_used_gb"] = torch.cuda.memory_allocated(cuda_id) / (1024 ** 3)
                memory_stats["cuda_available_gb"] = torch.cuda.get_device_properties(cuda_id).total_memory / (1024 ** 3) - memory_stats["cuda_used_gb"]
            except Exception as e:
                logger.warning(f"Could not get CUDA memory stats: {str(e)}")
                memory_stats["cuda_used_gb"] = 0
                memory_stats["cuda_available_gb"] = 0
        
        return memory_stats
    
    def estimate_optimal_batch_size(self, sample_text: str = "This is a sample text for batch size estimation", 
                                   max_batch_size: int = 128, 
                                   memory_threshold: float = 0.8) -> int:
        """
        Estimate the optimal batch size based on available memory.
        
        Args:
            sample_text (str): A representative text to use for estimation
            max_batch_size (int): Maximum batch size to consider
            memory_threshold (float): Memory usage threshold (0.0-1.0)
            
        Returns:
            int: Estimated optimal batch size
        """
        logger.info("Estimating optimal batch size based on available memory")
        
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Try with a single text first to measure per-item memory usage
        try:
            # Tokenize the sample text
            inputs = self.tokenizer(
                [sample_text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run a single prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get memory usage after single prediction
            single_memory = self.get_memory_usage()
            
            # Clean up
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            # Calculate memory used per item
            if torch.cuda.is_available():
                memory_per_item = (single_memory["cuda_used_gb"] - initial_memory["cuda_used_gb"])
                available_memory = single_memory["cuda_available_gb"]
            else:
                memory_per_item = (single_memory["system_used_gb"] - initial_memory["system_used_gb"])
                available_memory = single_memory["system_available_gb"]
            
            # Add safety margin and calculate optimal batch size
            if memory_per_item > 0:
                estimated_batch_size = int((available_memory * memory_threshold) / memory_per_item)
                optimal_batch_size = min(max(1, estimated_batch_size), max_batch_size)
                logger.info(f"Estimated memory per item: {memory_per_item:.4f} GB")
                logger.info(f"Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
            else:
                logger.warning("Could not measure per-item memory usage, defaulting to batch size of 32")
                return 32
                
        except Exception as e:
            logger.error(f"Error estimating batch size: {str(e)}")
            logger.warning("Defaulting to conservative batch size of 16")
            return 16
    
    def predict_batch(self, texts: List[str], batch_size: Optional[int] = None, 
                     show_progress: bool = True, auto_adjust_batch: bool = True) -> List[Dict[str, Any]]:
        """
        Predict emotions for a batch of texts with memory-efficient processing.

        Args:
            texts (list): List of text strings to analyze
            batch_size (int, optional): Maximum batch size to process at once.
                                       If None, will automatically determine an optimal size.
            show_progress (bool): Whether to show a progress bar for batch processing
            auto_adjust_batch (bool): Whether to automatically adjust batch size based on memory

        Returns:
            list: List of dictionaries containing emotion probabilities for each text
            
        Raises:
            ValueError: If texts is empty or contains invalid elements
            RuntimeError: If batch processing fails
        """
        # Validate input
        if not texts:
            logger.error("Invalid input: texts list is empty")
            raise ValueError("texts list must not be empty")
            
        # Filter out invalid texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered {len(texts) - len(valid_texts)} invalid entries from the texts list")
            
        if not valid_texts:
            logger.error("No valid texts found after filtering")
            raise ValueError("No valid texts to process after filtering")
            
        try:
            total_texts = len(valid_texts)
            logger.info(f"Processing batch of {total_texts} texts")
            
            # Auto-determine batch size if not specified or if auto-adjust enabled
            if batch_size is None or (auto_adjust_batch and batch_size > 16):
                batch_size = self.estimate_optimal_batch_size(
                    sample_text=valid_texts[0],
                    max_batch_size=min(128, total_texts)
                )
            
            # Process in smaller batches 
            if total_texts > batch_size:
                logger.info(f"Using batch size of {batch_size} for {total_texts} texts")
                results = []
                
                # Create progress bar if requested
                iterator = range(0, total_texts, batch_size)
                if show_progress:
                    iterator = tqdm.tqdm(iterator, total=(total_texts-1)//batch_size + 1, 
                                        desc="Processing batches", unit="batch")
                
                for i in iterator:
                    # Get the current batch
                    batch_texts = valid_texts[i:i+batch_size]
                    batch_size_actual = len(batch_texts)
                    
                    # Log batch information
                    batch_num = i//batch_size + 1
                    total_batches = (total_texts-1)//batch_size + 1
                    logger.debug(f"Processing batch {batch_num}/{total_batches} with {batch_size_actual} texts")
                    
                    # Process the batch
                    start_time = time.time()
                    batch_results = self._process_batch(batch_texts)
                    results.extend(batch_results)
                    
                    # Log performance metrics
                    elapsed = time.time() - start_time
                    texts_per_second = batch_size_actual / elapsed if elapsed > 0 else 0
                    logger.debug(f"Batch {batch_num} completed in {elapsed:.2f}s ({texts_per_second:.2f} texts/sec)")
                    
                    # Check memory usage and adjust batch size if needed
                    # Check memory usage and adjust batch size if needed
                    if auto_adjust_batch and i + batch_size < total_texts:
                        memory = self.get_memory_usage()
                        
                        # Check if we're running low on memory
                        if torch.cuda.is_available():
                            memory_used_pct = 1.0 - (memory["cuda_available_gb"] / 
                                                    (memory["cuda_used_gb"] + memory["cuda_available_gb"]))
                            threshold = 0.85  # Reduce batch size if using more than 85% of memory
                            
                            if memory_used_pct > threshold:
                                new_batch_size = max(1, int(batch_size * 0.8))  # Reduce by 20%
                                logger.warning(f"GPU memory usage high ({memory_used_pct:.1%}). "
                                             f"Reducing batch size from {batch_size} to {new_batch_size}")
                                batch_size = new_batch_size
                        else:
                            memory_used_pct = memory["system_percent"] / 100.0
                            threshold = 0.75  # Reduce batch size if using more than 75% of system memory
                            
                            if memory_used_pct > threshold:
                                new_batch_size = max(1, int(batch_size * 0.8))  # Reduce by 20%
                                logger.warning(f"System memory usage high ({memory_used_pct:.1%}). "
                                             f"Reducing batch size from {batch_size} to {new_batch_size}")
                                batch_size = new_batch_size
                    
                    # Clean up to prevent memory leaks
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return results
            else:
                # If the total texts is less than or equal to batch_size, process all at once
                return self._process_batch(valid_texts)
                
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def _process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Internal method to process a batch of texts.
        
        Args:
            texts (list): List of validated text strings to analyze
            
        Returns:
            list: List of dictionaries containing emotion probabilities for each text
        """
        # Enable gradient checkpointing if available (for memory efficiency)
        if hasattr(self.model, "gradient_checkpointing_enable") and len(texts) > 16:
            # Only used during inference here, but helps with memory
            self.model.gradient_checkpointing_enable()
        
        try:
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
                    'dominant_emotion': EMOTION_CLASSES[np.argmax(probs[i])],
                    'model_version': self.metadata.get('version', MODEL_VERSION)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise RuntimeError(f"Error processing batch: {str(e)}")
        finally:
            # Clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def save_model(self, save_path: str, version: Optional[str] = None):
        """
        Save the model to disk.

        Args:
            save_path (str): Path to save the model to
            version (str, optional): Version to assign to the saved model.
                                     If None, uses the current version.
                                     
        Returns:
            str: Path where the model was saved
            
        Raises:
            OSError: If the model cannot be saved to disk
        """
        try:
            # Create directory if it doesn't exist
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Update metadata with new version if provided
            if version:
                self.metadata['version'] = version
                
            # Update last_updated timestamp
            self.metadata['last_updated'] = datetime.now().isoformat()
            
            # Save model and tokenizer
            logger.info(f"Saving model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save metadata
            metadata_path = path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Model saved successfully to {path} with version {self.metadata['version']}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {str(e)}")
            raise OSError(f"Failed to save model: {str(e)}")
            
    def create_checkpoint(self, checkpoint_dir: str):
        """
        Create a checkpoint of the current model state.
        
        Args:
            checkpoint_dir (str): Base directory for checkpoints
            
        Returns:
            str: Path to the created checkpoint
            
        Raises:
            OSError: If the checkpoint cannot be created
        """
        try:
            # Create checkpoint directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{timestamp}"
            
            # Save checkpoint with metadata
            checkpoint_metadata = self.metadata.copy()
            checkpoint_metadata["checkpoint_time"] = timestamp
            checkpoint_metadata["checkpoint_type"] = "auto"
            
            # Create the directory
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            logger.info(f"Creating checkpoint at {checkpoint_path}")
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save checkpoint metadata
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)
                
            logger.info(f"Checkpoint created successfully at {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint in {checkpoint_dir}: {str(e)}")
            raise OSError(f"Failed to create checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory
            
        Returns:
            bool: True if checkpoint was loaded successfully
            
        Raises:
            OSError: If the checkpoint cannot be loaded
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Load the model from the checkpoint
            checkpoint_path = Path(checkpoint_path)
            
            # Check if checkpoint exists
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found at {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
                
            # Load metadata if available
            metadata_path = checkpoint_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded checkpoint metadata. Version: {self.metadata.get('version', 'unknown')}")
            
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise OSError(f"Failed to load checkpoint: {str(e)}")
