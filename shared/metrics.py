"""
Metrics module for the Exoplanet Classification project.
Calculates classification performance metrics and finds optimal thresholds.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss, roc_curve
)

def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all relevant classification metrics.
    
    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1)
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold (default 0.5)
        
    Returns:
        dict: All computed metrics
    """
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        "log_loss": log_loss(y_true, y_pred_proba)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    metrics["confusion_matrix"] = cm.tolist()
    
    # ROC AUC (handle constant predictions)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["roc_auc"] = 0.5 # Baseline for constant predictions
        
    return metrics

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal threshold using Youden's J statistic (TPR - FPR).
    
    Returns:
        float: optimal threshold
        float: max J statistic
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    except ValueError:
        return 0.5, 0.0

def save_metrics(metrics_dict, algo_name, results_dir):
    """Save metrics to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{algo_name}_metrics.json")
    
    # Convert numpy types to native for JSON
    clean_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v
            
    with open(file_path, 'w') as f:
        json.dump(clean_metrics, f, indent=4)
    print(f"  [Metrics] Sauvegardées dans {file_path}")

def save_predictions(y_pred_proba, algo_name, results_dir, threshold=0.5):
    """Save predictions on unknown data."""
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{algo_name}_predictions.csv")
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    df = pd.DataFrame({
        "index": np.arange(len(y_pred_proba)),
        "probability": y_pred_proba,
        "prediction": y_pred_binary
    })
    
    df.to_csv(file_path, index=False)
    print(f"  [Preds] Sauvegardées dans {file_path}")

def print_metrics_summary(metrics, algo_name):
    """Print formatted summary of metrics."""
    print(f"\n--- RÉSULTATS : {algo_name} ---")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC AUC   : {metrics['roc_auc']:.4f}")
    print("-" * (18 + len(algo_name)))
