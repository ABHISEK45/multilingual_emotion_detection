def test_compute_metrics():
    """Test computation of evaluation metrics."""
    # Perfect predictions
    y_true = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
    y_pred = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Check that all metrics are computed
    assert 'accuracy' in metrics
    assert 'precision_macro' in metrics
    assert 'recall_macro' in metrics
    assert 'f1_macro' in metrics
    
    # For perfect predictions, all metrics should be 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['precision_macro'] == 1.0
    assert metrics['recall_macro'] == 1.0
    assert metrics['f1_macro'] == 1.0
    
    # Check per-class metrics
    assert 'per_class' in metrics
    for emotion in EMOTION_CLASSES:
        assert emotion in metrics['per_class']
        assert metrics['per_class'][emotion]['precision'] == 1.0
        assert metrics['per_class'][emotion]['recall'] == 1.0
        assert metrics['per_class'][emotion]['f1'] == 1.0
    
    # Imperfect predictions
    y_true = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
    y_pred = ["Joy", "Joy", "Anger", "Fear", "Neutral", "Neutral"]
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Only 4 out of 6 correct, accuracy should be 0.6667
    assert np.isclose(metrics['accuracy'], 4/6)
    
    # Check that per-class metrics reflect correct predictions
    assert metrics['per_class']["Joy"]['precision'] == 0.5  # 1 correct out of 2 predicted
    assert metrics['per_class']["Sadness"]['recall'] == 0.0  # 0 correctly predicted out of 1 actual


def test_generate_confusion_matrix():
    """Test generation of confusion matrix."""
    # Test with perfect predictions
    y_true = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
    y_pred = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
    
    cm = generate_confusion_matrix(y_true, y_pred)
    
    # Check dimensions
    assert cm.shape == (6, 6)  # 6x6 for 6 emotion classes
    
    # For perfect predictions, should have 1s on the diagonal and 0s elsewhere
    np.testing.assert_array_equal(np.diag(cm), np.ones(6))
    assert np.sum(cm) == 6  # Sum of all elements should be 6
    
    # Test normalization
    cm_normalized = generate_confusion_matrix(y_true, y_pred, normalize=True)
    
    # Normalized matrix should have 1.0s on the diagonal
    np.testing.assert_array_equal(np.diag(cm_normalized), np.ones(6))
    
    # All rows should sum to 1
    for i in range(6):
        assert np.isclose(np.sum(cm_normalized[i, :]), 1.0)
    
    # Test with imbalanced classes
    y_true = ["Joy", "Joy", "Joy", "Sadness"]
    y_pred = ["Joy", "Joy", "Sadness", "Sadness"]
    
    cm = generate_confusion_matrix(y_true, y_pred)
    
    # Should have specific values based on the predictions
    assert cm[0, 0] == 2  # True Joy, Predicted Joy
    assert cm[0, 1] == 1  # True Joy, Predicted Sadness
    assert cm[1, 1] == 1  # True Sadness, Predicted Sadness


def test_plot_confusion_matrix(test_data_dir):
    """Test plotting confusion matrix."""
    # Create a simple confusion matrix
    cm = np.array([
        [2, 1, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    # Test with default parameters
    fig = plot_confusion_matrix(cm)
    assert isinstance(fig, plt.Figure)
    
    # Test saving the plot
    output_path = test_data_dir / "test_confusion_matrix.png"
    fig = plot_confusion_matrix(cm, output_path=output_path)
    assert output_path.exists()
    
    # Test with normalization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plot_confusion_matrix(cm_normalized, normalize=True)
    assert isinstance(fig, plt.Figure)
    
    plt.close('all')  # Clean up


def test_create_evaluation_report(test_data_dir):
    """Test creating evaluation report."""
    # Create sample metrics
    metrics = {
        'accuracy': 0.85,
        'precision_macro': 0.82,
        'recall_macro': 0.80,
        'f1_macro': 0.81,
        'precision_weighted': 0.84,
        'recall_weighted': 0.85,
        'f1_weighted': 0.84,
        'per_class': {
            'Joy': {'precision': 0.90, 'recall': 0.85, 'f1': 0.87},
            'Sadness': {'precision': 0.80, 'recall': 0.82, 'f1': 0.81},
            'Anger': {'precision': 0.75, 'recall': 0.70, 'f1': 0.72},
            'Fear': {'precision': 0.85, 'recall': 0.78, 'f1': 0.81},
            'Surprise': {'precision': 0.82, 'recall': 0.80, 'f1': 0.81},
            'Neutral': {'precision': 0.88, 'recall': 0.90, 'f1': 0.89}
        }
    }
    
    # Test report generation
    report = create_evaluation_report(metrics)
    
    # Check that the report is a string
    assert isinstance(report, str)
    
    # Check that key metrics are included in the report
    assert "Accuracy: 0.8500" in report
    assert "Precision: 0.8200" in report
    assert "Recall: 0.8000" in report
    
    # Check that per-class metrics are included
    for emotion in EMOTION_CLASSES:
        assert f"### {emotion}" in report
    
    # Test saving report to file
    output_path = test_data_dir / "test_evaluation_report.md"
    report = create_evaluation_report(metrics, output_path=output_path)
    assert output_path.exists()
    
    # Check file contents
    with open(output_path, 'r') as f:
        file_content = f.read()
    assert file_content == report


def test_create_results_dataframe(sample_texts, sample_labels, prediction_results):
    """Test creating results dataframe."""
    # Use a subset of the fixtures for consistent lengths
    texts = sample_texts[:4]
    true_emotions = sample_labels[:4]
    
    # Test dataframe creation
    df = create_results_dataframe(texts, true_emotions, prediction_results)
    
    # Check dataframe structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(texts)
    assert 'text' in df.columns
    assert 'true_emotion' in df.columns
    assert 'predicted_emotion' in df.columns
    assert 'correct' in df.columns
    
    # Check for emotion probability columns
    for emotion in EMOTION_CLASSES:
        assert f"{emotion}_prob" in df.columns
    
    # Check that the correct column is calculated properly
    for i, row in df.iterrows():
        assert row['correct'] == (row['true_emotion'] == row['predicted_emotion'])


def test_evaluate(sample_dataset, prediction_results, test_data_dir):
    """Test evaluation pipeline."""
    # Use a subset of the dataset to match prediction_results length
    df = sample_dataset.iloc[:4].copy()
    
    # Test basic evaluation
    eval_results = evaluate(prediction_results, df, 'text', 'emotion')
    
    # Check structure of evaluation results
    assert 'metrics' in eval_results
    assert 'confusion_matrix' in eval_results
    assert 'confusion_matrix_normalized' in eval_results
    assert 'results_df' in eval_results
    
    # Check metrics calculation
    assert 'accuracy' in eval_results['metrics']
    
    # Check confusion matrix dimensions
    assert eval_results['confusion_matrix'].shape == (len(EMOTION_CLASSES), len(EMOTION_CLASSES))
    
    # Test evaluation with output directory
    output_dir = test_data_dir / "evaluation_output"
    eval_results = evaluate(prediction_results, df, 'text', 'emotion', output_dir=output_dir)
    
    # Check that output files are created
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "confusion_matrix_normalized.png").exists()
    assert (output_dir / "evaluation_report.md").exists()
    assert (output_dir / "prediction_results.csv").exists()
    
    # Test column inference
    # Rename columns to test inference
    df_renamed = df.rename(columns={'text': 'input_text', 'emotion': 'emotion_label'})
    
    # This should raise an error because it can't infer text column
    with pytest.raises(ValueError):
        evaluate(prediction_results, df_renamed.drop(columns=['input_text']), label_column='emotion_label')
    
    # This should raise an error because it can't infer label column
    with pytest.raises(ValueError):
        evaluate(prediction_results, df_renamed.drop(columns=['emotion_label']), text_column='input_text')


def test_plot_emotion_distribution(sample_dataset, test_data_dir):
    """Test plotting emotion distribution."""
    # Test basic plot generation
    fig = plot_emotion_distribution(sample_dataset, 'emotion')
    assert isinstance(fig, plt.Figure)
    
    # Test saving the plot
    output_path = test_data_dir / "emotion_distribution.png"
    fig = plot_emotion_distribution(sample_dataset, 'emotion', output_path=output_path)
    assert output_path.exists()
    
    plt.close('all')  # Clean up


def test_plot_top_misclassifications(test_data_dir):
    """Test plotting top misclassifications."""
    # Create sample results dataframe with some misclassifications
    results_df = pd.DataFrame({
        'text': [f"Sample text {i}" for i in range(10)],
        'true_emotion': ['Joy', 'Joy', 'Sadness', 'Sadness', 'Anger', 'Anger', 'Fear', 'Fear', 'Surprise', 'Neutral'],
        'predicted_emotion': ['Joy', 'Sadness', 'Sadness', 'Joy', 'Anger', 'Fear', 'Fear', 'Anger', 'Neutral', 'Surprise']
    })
    
    # Test basic plot generation
    fig = plot_top_misclassifications(results_df, n=3)
    assert isinstance(fig, plt.Figure)
    
    # Test saving the plot
    output_path = test_data_dir / "top_misclassifications.png"
    fig = plot_top_misclassifications(results_df, n=3, output_path=output_path)
    assert output_path.exists()
    
    # Test with no misclassifications
    perfect_df = pd.DataFrame({
        'text': ["Perfect 1", "Perfect 2"],
        'true_emotion': ['Joy', 'Sadness'],
        'predicted_emotion': ['Joy', 'Sadness']
    })
    
    # This should not raise an error, but produce an empty plot
    fig = plot_top_misclassifications(perfect_df)
    assert isinstance(fig, plt.Figure)
    
    plt.close('all')  # Clean up


def test_load_model():
    """Test load_model function (wrapper to avoid circular imports)."""
    # This is mostly a smoke test to ensure the function exists
    # Actual model loading is tested in test_model.py
    from src.evaluation import load_model
    
    # Check that the function exists and is callable
    assert callable(load_model)

"""
Tests for the evaluation module.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.evaluation import (
    compute_metrics,
    generate_confusion_matrix,
    plot_confusion_matrix,
    create_evaluation_report,
    create_results_dataframe,
    evaluate,
    plot_emotion_distribution,
    plot_top_misclassifications
)
from src.model import EMOTION_CLASSES

