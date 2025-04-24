"""
Evaluation module for multilingual emotion detection.

This module provides functions for evaluating the performance of emotion detection models,
including metric calculations, confusion matrix generation, and result visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import emotion classes from the model module
from .model import EMOTION_CLASSES


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for emotion detection.

    Args:
        y_true (list): Ground truth emotion labels
        y_pred (list): Predicted emotion labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        # Convert to binary classification problem for each emotion
        y_true_binary = [1 if y == emotion else 0 for y in y_true]
        y_pred_binary = [1 if y == emotion else 0 for y in y_pred]
        
        # Compute metrics
        class_metrics[emotion] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }
    
    metrics['per_class'] = class_metrics
    
    return metrics


def generate_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Generate a confusion matrix for emotion detection results.

    Args:
        y_true (list): Ground truth emotion labels
        y_pred (list): Predicted emotion labels
        normalize (bool, optional): Whether to normalize the confusion matrix.
                                   Defaults to False.

    Returns:
        numpy.ndarray: Confusion matrix
    """
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_CLASSES)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def plot_confusion_matrix(cm, output_path=None, figsize=(10, 8), normalize=False, title=None):
    """
    Plot a confusion matrix.

    Args:
        cm (numpy.ndarray): Confusion matrix
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
                                    Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 8).
        normalize (bool, optional): Whether the confusion matrix is normalized.
                                   Defaults to False.
        title (str, optional): Plot title. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set title
    if title is None:
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(title, fontsize=16)
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=EMOTION_CLASSES,
        yticklabels=EMOTION_CLASSES
    )
    
    # Set labels
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()


def create_evaluation_report(metrics, output_path=None):
    """
    Create a comprehensive evaluation report in text format.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_path (str, optional): Path to save the report. If None, the report is returned.
                                    Defaults to None.

    Returns:
        str: Evaluation report as a formatted string
    """
    # Create the report
    report = ["# Emotion Detection Evaluation Report\n"]
    
    # Overall metrics
    report.append("## Overall Metrics\n")
    report.append(f"Accuracy: {metrics['accuracy']:.4f}\n")
    report.append("\n### Macro-averaged Metrics\n")
    report.append(f"Precision: {metrics['precision_macro']:.4f}\n")
    report.append(f"Recall: {metrics['recall_macro']:.4f}\n")
    report.append(f"F1 Score: {metrics['f1_macro']:.4f}\n")
    
    report.append("\n### Weighted-averaged Metrics\n")
    report.append(f"Precision: {metrics['precision_weighted']:.4f}\n")
    report.append(f"Recall: {metrics['recall_weighted']:.4f}\n")
    report.append(f"F1 Score: {metrics['f1_weighted']:.4f}\n")
    
    # Per-class metrics
    report.append("\n## Per-class Metrics\n")
    for emotion, class_metrics in metrics['per_class'].items():
        report.append(f"\n### {emotion}\n")
        report.append(f"Precision: {class_metrics['precision']:.4f}\n")
        report.append(f"Recall: {class_metrics['recall']:.4f}\n")
        report.append(f"F1 Score: {class_metrics['f1']:.4f}\n")
    
    # Join the report
    report_str = "".join(report)
    
    # Save or return the report
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_str)
    
    return report_str


def create_results_dataframe(texts, true_emotions, predicted_results):
    """
    Create a DataFrame containing prediction results and ground truth.

    Args:
        texts (list): List of input texts
        true_emotions (list): List of ground truth emotion labels
        predicted_results (list): List of prediction result dictionaries

    Returns:
        pandas.DataFrame: DataFrame containing results
    """
    # Extract predicted emotions
    pred_emotions = [result['dominant_emotion'] for result in predicted_results]
    
    # Create emotion probability columns
    emotion_probs = {}
    for emotion in EMOTION_CLASSES:
        emotion_probs[f"{emotion}_prob"] = [
            result['emotions'].get(emotion, 0) for result in predicted_results
        ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'true_emotion': true_emotions,
        'predicted_emotion': pred_emotions,
        'correct': [true == pred for true, pred in zip(true_emotions, pred_emotions)],
        **emotion_probs
    })
    
    return df


def evaluate(prediction_results, dataset, text_column=None, label_column=None, output_dir=None):
    """
    Evaluate emotion detection model performance.

    Args:
        prediction_results (list): List of prediction result dictionaries
        dataset (pandas.DataFrame): Dataset containing ground truth
        text_column (str, optional): Name of the column containing text.
                                    If None, tries to infer. Defaults to None.
        label_column (str, optional): Name of the column containing emotion labels.
                                     If None, tries to infer. Defaults to None.
        output_dir (str, optional): Directory to save evaluation outputs.
                                  If None, outputs are not saved. Defaults to None.

    Returns:
        dict: Dictionary containing evaluation results and metrics
    """
    # Try to infer column names if not provided
    if text_column is None:
        text_columns = [col for col in dataset.columns if 'text' in col.lower()]
        if text_columns:
            text_column = text_columns[0]
        else:
            raise ValueError("Could not infer text column. Please specify 'text_column'.")
    
    if label_column is None:
        label_columns = [
            col for col in dataset.columns 
            if any(word in col.lower() for word in ['emotion', 'label', 'sentiment'])
        ]
        if label_columns:
            label_column = label_columns[0]
        else:
            raise ValueError("Could not infer label column. Please specify 'label_column'.")
    
    # Extract data
    texts = dataset[text_column].tolist()
    true_emotions = dataset[label_column].tolist()
    pred_emotions = [result['dominant_emotion'] for result in prediction_results]
    
    # Compute metrics
    metrics = compute_metrics(true_emotions, pred_emotions)
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(true_emotions, pred_emotions)
    cm_normalized = generate_confusion_matrix(true_emotions, pred_emotions, normalize=True)
    
    # Create results DataFrame
    results_df = create_results_dataframe(texts, true_emotions, prediction_results)
    
    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save plots
        plot_confusion_matrix(
            cm, 
            output_path=output_path / "confusion_matrix.png", 
            title="Confusion Matrix"
        )
        plot_confusion_matrix(
            cm_normalized, 
            output_path=output_path / "confusion_matrix_normalized.png", 
            normalize=True,
            title="Normalized Confusion Matrix"
        )
        
        # Save report
        create_evaluation_report(metrics, output_path=output_path / "evaluation_report.md")
        
        # Save results
        results_df.to_csv(output_path / "prediction_results.csv", index=False)
    
    # Return evaluation results
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'results_df': results_df
    }


def plot_emotion_distribution(data, emotion_column, output_path=None, figsize=(10, 6)):
    """
    Plot the distribution of emotions in a dataset.

    Args:
        data (pandas.DataFrame): Dataset containing emotion labels
        emotion_column (str): Name of the column containing emotion labels
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
                                   Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 6).

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Count emotions
    emotion_counts = data[emotion_column].value_counts()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    
    # Add labels and title
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Emotion Distribution', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()


def plot_top_misclassifications(results_df, n=10, output_path=None, figsize=(12, 8)):
    """
    Plot the top misclassified emotion pairs.

    Args:
        results_df (pandas.DataFrame): DataFrame containing prediction results
        n (int, optional): Number of top misclassification pairs to plot. Defaults to 10.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
                                   Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to (12, 8).

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get misclassified samples
    misclassified = results_df[results_df['true_emotion'] != results_df['predicted_emotion']]
    
    # Count misclassification pairs
    pairs = misclassified.groupby(['true_emotion', 'predicted_emotion']).size().reset_index()
    pairs.columns = ['True Emotion', 'Predicted Emotion', 'Count']
    
    # Sort and take top N
    pairs = pairs.sort_values('Count', ascending=False).head(n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    ax = sns.barplot(x='Count', y='True Emotion', hue='Predicted Emotion', data=pairs)
    
    # Add labels and title
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.title(f'Top {n} Misclassification Pairs', fontsize=16)
    
    # Adjust legend
    plt.legend(title='Predicted Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return plt.gcf()


def load_model(model_path=None):
    """
    Wrapper function to load the emotion detection model.
    
    This function exists in the evaluation module to avoid circular imports.

    Args:
        model_path (str, optional): Path to the model. Defaults to None.

    Returns:
        EmotionDetectionModel: Loaded model
    """
    from .model import EmotionDetectionModel
    return EmotionDetectionModel(model_path)

