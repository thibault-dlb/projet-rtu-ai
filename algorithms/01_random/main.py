"""
Random Baseline Algorithm for Exoplanet Classification.
Generates random probabilities and serves as a performance floor.
"""
import os
import sys
import numpy as np

# Set path to root for shared module access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.config import (
    SEED, DEFAULT_THRESHOLD, METRICS_DIR, PREDICTIONS_DIR
)
from shared.data_loader import load_and_prepare_data
from shared.metrics import (
    compute_all_metrics, find_optimal_threshold, 
    save_metrics, save_predictions, print_metrics_summary
)
from shared.resource_monitor import ResourceMonitor, save_resource_stats

def main():
    np.random.seed(SEED)
    data = load_and_prepare_data()
    
    with ResourceMonitor() as monitor:
        # Random probabilities for test set
        y_test_proba = np.random.rand(len(data.y_test))
        metrics = compute_all_metrics(data.y_test, y_test_proba, DEFAULT_THRESHOLD)
        opt_threshold, j_stat = find_optimal_threshold(data.y_test, y_test_proba)
        
        # Predictions on unknown data
        y_unknown_proba = np.random.rand(len(data.X_unknown))
        
        # Save
        save_metrics(metrics, "01_random", METRICS_DIR)
        save_resource_stats(monitor.get_stats(), "01_random", METRICS_DIR)
        save_predictions(y_unknown_proba, "01_random", PREDICTIONS_DIR, threshold=opt_threshold)
        
        print_metrics_summary(metrics, "Aléatoire")
        print(f"Seuil optimal suggéré : {opt_threshold:.4f} (J={j_stat:.4f})")

if __name__ == "__main__":
    main()
