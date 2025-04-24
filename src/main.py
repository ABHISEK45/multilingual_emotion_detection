"""
Main entry point for the multilingual emotion detection system.

This module orchestrates the process of loading models, preprocessing text,
performing emotion detection, and evaluating results.
"""

import argparse
import os
from pathlib import Path

from . import preprocessor
from . import model
from . import evaluation


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Multilingual Emotion Detection')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--language', type=str, default='auto', 
                        choices=['auto', 'en', 'hi'], 
                        help='Language of the text (auto, en, hi)')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join('models', 'emotion_model'),
                        help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for processing')
    parser.add_argument('--dataset', type=str, 
                        help='Path to dataset for batch processing')
    parser.add_argument('--output', type=str, 
                        help='Path to save results')
    
    return parser.parse_args()


def detect_emotion(text, language='auto', model_path=None):
    """
    Detect emotion in a given text.

    Args:
        text (str): Input text
        language (str, optional): Language code ('en', 'hi', or 'auto' for detection). 
                                 Defaults to 'auto'.
        model_path (str, optional): Path to the model. Defaults to None.

    Returns:
        dict: Dictionary containing detected emotions and their probabilities
    """
    # Detect language if set to auto
    if language == 'auto':
        language = preprocessor.detect_language(text)
    
    # Preprocess text based on language
    processed_text = preprocessor.preprocess_text(text, language)
    
    # Load model
    emotion_model = model.load_model(model_path)
    
    # Get emotion predictions
    predictions = emotion_model.predict(processed_text)
    
    return predictions


def batch_process(dataset_path, output_path, model_path=None, batch_size=16):
    """
    Process a dataset in batch mode.

    Args:
        dataset_path (str): Path to the dataset
        output_path (str): Path to save results
        model_path (str, optional): Path to the model. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 16.

    Returns:
        dict: Evaluation metrics
    """
    # Load dataset
    data = preprocessor.load_dataset(dataset_path)
    
    # Load model
    emotion_model = model.load_model(model_path)
    
    # Process in batches
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_results = emotion_model.predict_batch(batch)
        results.extend(batch_results)
    
    # Evaluate results
    metrics = evaluation.evaluate(results, data)
    
    # Save results
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for item in results:
                f.write(f"{item}\n")
    
    return metrics


def main():
    """Main function to execute when script is run."""
    args = parse_arguments()
    
    if args.text:
        result = detect_emotion(args.text, args.language, args.model_path)
        print(f"Detected emotions: {result}")
    elif args.dataset:
        metrics = batch_process(
            args.dataset, 
            args.output, 
            args.model_path, 
            args.batch_size
        )
        print(f"Evaluation metrics: {metrics}")
    else:
        print("Error: Please provide either --text or --dataset")


if __name__ == "__main__":
    main()

