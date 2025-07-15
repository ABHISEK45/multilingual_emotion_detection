"""
Main entry point for the multilingual emotion detection system.

This module orchestrates the process of loading models, preprocessing text,
performing emotion detection, and evaluating results.
"""

import argparse
import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from tqdm import tqdm
import torch

from . import preprocessor
from . import model
from . import evaluation

from . import config
from . import logger

# Create logger
logger = logger.get_logger(__name__)


def validate_args(args) -> Tuple[bool, str]:
    """
    Validate command line arguments.
    
    Args:
        args: The parsed command line arguments
        
    Returns:
        tuple: (is_valid, error_message) where is_valid is a boolean and
               error_message is a string describing the error (if any)
    """
    # Check if required arguments are provided
    if not args.text and not args.dataset:
        return False, "Either --text or --dataset must be provided"
    
    # Validate text input if provided
    if args.text and not isinstance(args.text, str):
        return False, "Text argument must be a string"
    
    # Validate language if provided
    if args.language not in ['auto', 'en', 'hi']:
        return False, f"Invalid language: {args.language}. Must be one of: auto, en, hi"
    
    # Validate model path if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists() and not model_path.parent.exists():
            return False, f"Model path does not exist and cannot be created: {args.model_path}"
    
    # Validate dataset path if provided
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            return False, f"Dataset file not found: {args.dataset}"
        
        # Check if file extension is supported
        if dataset_path.suffix.lower() not in ['.csv', '.json', '.xlsx', '.xls']:
            return False, f"Unsupported dataset format: {dataset_path.suffix}. Supported formats: .csv, .json, .xlsx, .xls"
    
    # Validate batch size if provided
    if args.batch_size <= 0:
        return False, f"Batch size must be positive, got {args.batch_size}"
    
    # Validate output path if provided
    if args.output:
        output_path = Path(args.output)
        # Check if parent directory exists or can be created
        if not output_path.parent.exists():
            try:
                # Don't actually create it, just check if we can
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create output directory for {args.output}: {str(e)}"
    
    return True, ""


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Multilingual Emotion Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows default values
    )
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--language', type=str, default='auto', 
                        choices=['auto', 'en', 'hi'], 
                        help='Language of the text (auto, en, hi)')
    parser.add_argument('--model_path', type=str, 
                        default=config.get_config().get("MODEL", "PATH"),
                        help='Path to the model')
    parser.add_argument('--batch_size', type=int, 
                        default=config.get_config().get("PROCESSING", "DEFAULT_BATCH_SIZE"), 
                        help='Batch size for processing')
    parser.add_argument('--dataset', type=str, 
                        help='Path to dataset for batch processing')
    parser.add_argument('--output', type=str, 
                        help='Path to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    args = parser.parse_args()
    
    # Set log level based on argument
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate arguments
    is_valid, error_message = validate_args(args)
    if not is_valid:
        parser.error(error_message)
    
    return args


def detect_emotion(text: str, language: str = 'auto', model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect emotion in a given text.

    Args:
        text (str): Input text
        language (str, optional): Language code ('en', 'hi', or 'auto' for detection). 
                                 Defaults to 'auto'.
        model_path (str, optional): Path to the model. Defaults to None.

    Returns:
        dict: Dictionary containing detected emotions and their probabilities
        
    Raises:
        ValueError: If input text is invalid or language is not supported
        RuntimeError: If model loading or prediction fails
    """
    try:
        logger.info(f"Processing text for emotion detection. Language: {language}")
        
        # Validate input text
        if not text or not isinstance(text, str):
            logger.error("Invalid input: Text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")
        
        # Detect language if set to auto
        if language == 'auto':
            logger.debug("Auto-detecting language")
            try:
                language = preprocessor.detect_language(text)
                logger.info(f"Detected language: {language}")
            except Exception as e:
                logger.error(f"Language detection failed: {str(e)}")
                raise RuntimeError(f"Language detection failed: {str(e)}")
        
        # Validate language
        if language not in ['en', 'hi']:
            logger.error(f"Unsupported language: {language}")
            raise ValueError(f"Unsupported language: {language}. Supported languages are: en, hi")
        
        # Preprocess text based on language
        logger.debug(f"Preprocessing text in {language}")
        try:
            processed_text = preprocessor.preprocess_text(text, language)
            logger.debug(f"Processed text: {processed_text[:50]}...")
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            raise RuntimeError(f"Text preprocessing failed: {str(e)}")
        
        # Load model
        logger.debug(f"Loading model from: {model_path if model_path else 'default'}")
        try:
            emotion_model = model.EmotionDetectionModel(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        # Get emotion predictions
        logger.debug("Predicting emotions")
        try:
            predictions = emotion_model.predict(processed_text)
            logger.info(f"Prediction complete. Dominant emotion: {predictions.get('dominant_emotion', 'unknown')}")
            return predictions
        except Exception as e:
            logger.error(f"Emotion prediction failed: {str(e)}")
            raise RuntimeError(f"Emotion prediction failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in detect_emotion: {str(e)}")
        raise


def batch_process(dataset_path: str, output_path: Optional[str] = None, 
                 model_path: Optional[str] = None, batch_size: int = 16,
                 show_progress: bool = True) -> Dict[str, Any]:
    """
    Process a dataset in batch mode with robust error handling and progress tracking.

    Args:
        dataset_path (str): Path to the dataset
        output_path (str, optional): Path to save results. If None, results won't be saved.
        model_path (str, optional): Path to the model. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 16.
        show_progress (bool, optional): Whether to show progress bars. Defaults to True.

    Returns:
        dict: Evaluation metrics
        
    Raises:
        FileNotFoundError: If dataset file is not found
        ValueError: If dataset format is invalid
        RuntimeError: If processing fails
    """
    start_time = time.time()
    logger.info(f"Starting batch processing of dataset: {dataset_path}")
    
    # Variables to track processing
    data = None
    emotion_model = None
    results = []
    metrics = {}
    total_texts = 0
    processed_texts = 0
    
    try:
        # Load dataset with proper error handling
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            data = preprocessor.load_dataset(dataset_path)
            total_texts = len(data)
            logger.info(f"Dataset loaded successfully. Total entries: {total_texts}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise ValueError(f"Failed to load dataset: {str(e)}")
            
        # Check if dataset is empty
        if total_texts == 0:
            logger.error("Dataset is empty")
            raise ValueError("Dataset is empty")
            
        # Identify text column
        text_column = None
        for column in data.columns:
            if column.lower() in ['text', 'content', 'message', 'review', 'comment']:
                text_column = column
                break
                
        if text_column is None:
            logger.error("Could not identify text column in dataset")
            raise ValueError("Could not identify text column in dataset. Expected column names: text, content, message, review, comment")
            
        # Load model with proper error handling
        logger.info(f"Loading model from {model_path if model_path else 'default'}")
        try:
            emotion_model = model.EmotionDetectionModel(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
        # Prepare dataset for processing
        logger.info("Preparing dataset for processing")
        try:
            # Extract texts for processing
            texts = data[text_column].tolist()
            
            # Filter out invalid entries
            valid_texts = [t for t in texts if t and isinstance(t, str)]
            if len(valid_texts) != len(texts):
                logger.warning(f"Filtered {len(texts) - len(valid_texts)} invalid entries from the dataset")
                
            if not valid_texts:
                logger.error("No valid texts found in dataset after filtering")
                raise ValueError("No valid texts found in dataset after filtering")
                
            # Update total count
            total_texts = len(valid_texts)
            logger.info(f"Dataset prepared. Processing {total_texts} valid entries")
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {str(e)}")
            raise ValueError(f"Failed to prepare dataset: {str(e)}")
            
        # Process in batches with progress tracking
        logger.info(f"Processing dataset with batch size: {batch_size}")
        
        # Process texts using the model's batch prediction
        try:
            results = emotion_model.predict_batch(
                valid_texts, 
                batch_size=batch_size,
                show_progress=show_progress,
                auto_adjust_batch=True
            )
            processed_texts = len(results)
            logger.info(f"Processed {processed_texts}/{total_texts} texts successfully")
        except Exception as e:
            logger.error(f"Failed during batch prediction: {str(e)}")
            raise RuntimeError(f"Failed during batch prediction: {str(e)}")
        
        # Evaluate results
        logger.info("Calculating evaluation metrics")
        try:
            if hasattr(data, 'get') and 'label' in data.columns:
                metrics = evaluation.evaluate(results, data)
                logger.info(f"Evaluation metrics calculated: {metrics}")
            else:
                logger.info("No ground truth labels found in dataset. Skipping evaluation.")
                metrics = {"processed_texts": processed_texts, "total_texts": total_texts}
        except Exception as e:
            logger.error(f"Failed to calculate evaluation metrics: {str(e)}")
            metrics = {"error": str(e), "processed_texts": processed_texts, "total_texts": total_texts}
        
        # Save results
        if output_path:
            logger.info(f"Saving results to {output_path}")
            try:
                _save_results(results, output_path)
                logger.info(f"Results saved successfully to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
                raise RuntimeError(f"Failed to save results: {str(e)}")
        
        # Log processing summary
        elapsed_time = time.time() - start_time
        processing_rate = total_processed = len(results)
        if elapsed_time > 0:
            processing_rate = total_processed / elapsed_time
        logger.info(f"Batch processing completed in {elapsed_time:.2f}s ({processing_rate:.2f} texts/sec)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        # Attempt to save any results collected so far
        if results and output_path:
            try:
                recovery_output = f"{output_path}.recovered"
                logger.info(f"Attempting to save {len(results)} processed results to {recovery_output}")
                _save_results(results, recovery_output)
                logger.info(f"Recovered results saved to {recovery_output}")
            except Exception as recovery_error:
                logger.error(f"Failed to save recovered results: {str(recovery_error)}")
        
        # Re-raise the exception
        raise
    finally:
        # Clean up resources
        logger.debug("Cleaning up resources")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Helper function to save results to disk.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save results to
        
    Raises:
        IOError: If saving fails
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file format
    file_ext = Path(output_path).suffix.lower()
    
    try:
        if file_ext == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif file_ext == '.csv':
            # Flatten the results for CSV
            import pandas as pd
            flattened = []
            for item in results:
                flat_item = {
                    'text': item.get('text', ''),
                    'dominant_emotion': item.get('dominant_emotion', '')
                }
                # Add individual emotion scores
                if 'emotions' in item:
                    for emotion, score in item['emotions'].items():
                        flat_item[f'emotion_{emotion}'] = score
                flattened.append(flat_item)
            pd.DataFrame(flattened).to_csv(output_path, index=False)
        else:
            # Default text output
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(f"{item}\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {str(e)}")


def main():
    """Main function to execute when script is run."""
    # Track start time for overall performance monitoring
    start_time = time.time()
    
    # Setup signal handlers for graceful exit
    import signal
    
    def signal_handler(sig, frame):
        logger.warning(f"Received signal {sig}. Exiting gracefully...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        logger.debug(f"Arguments parsed: {vars(args)}")
        
        # Process based on arguments
        if args.text:
            logger.info(f"Processing single text input: {args.text[:50]}...")
            try:
                result = detect_emotion(args.text, args.language, args.model_path)
                
                # Format and print result
                print("\nEmotion Analysis Results:")
                print(f"Text: {args.text}")
                print(f"Dominant Emotion: {result['dominant_emotion']}")
                print("\nEmotion Scores:")
                for emotion, score in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {score:.4f}")
                    
                logger.info(f"Successfully processed text with dominant emotion: {result['dominant_emotion']}")
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                print(f"\nError: {str(e)}")
                sys.exit(1)
                
        elif args.dataset:
            logger.info(f"Processing dataset: {args.dataset}")
            try:
                metrics = batch_process(
                    args.dataset, 
                    args.output, 
                    args.model_path, 
                    args.batch_size,
                    not args.no_progress  # Show progress unless --no-progress flag is set
                )
                
                # Format and print evaluation metrics
                print("\nDataset Processing Results:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
                    
                if args.output:
                    print(f"\nResults saved to: {args.output}")
                    
                logger.info(f"Successfully processed dataset with metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error processing dataset: {str(e)}")
                print(f"\nError: {str(e)}")
                sys.exit(1)
        else:
            logger.error("No input provided. Please specify either --text or --dataset")
            print("Error: Please provide either --text or --dataset")
            sys.exit(1)
            
        # Log execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Final cleanup
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


if __name__ == "__main__":
    main()
