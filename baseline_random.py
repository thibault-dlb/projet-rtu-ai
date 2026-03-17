"""
baseline_random.py
Random classifier — establishes a baseline (floor) score.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from data_loader import ExoplanetDataLoader


def random_classify(X, rng):
    """Predict 0 or 1 uniformly at random for each sample."""
    return rng.integers(0, 2, size=len(X))


def main():
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    loader.summary(X_train, X_test, y_train, y_test)

    rng = np.random.default_rng(seed=42)
    y_pred = random_classify(X_test, rng)

    print("\n--- RANDOM BASELINE ---")
    print(classification_report(y_test, y_pred, target_names=["Other", "Planet"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
