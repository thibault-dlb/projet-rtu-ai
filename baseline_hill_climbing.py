"""
baseline_hill_climbing.py
Hill Climbing classifier using a simple single-layer perceptron.

Strategy:
  - Start with random weights + bias.
  - Each iteration, mutate a copy by adding small gaussian noise.
  - Keep the mutant if its F1 score on the training set improves.
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from data_loader import ExoplanetDataLoader


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict(X, weights, bias, threshold=0.5):
    """Simple perceptron prediction."""
    z = X @ weights + bias
    return (sigmoid(z) >= threshold).astype(int)


def evaluate(X, y, weights, bias):
    """Return F1 score for the given weights."""
    y_pred = predict(X, weights, bias)
    return f1_score(y, y_pred, zero_division=0)


def hill_climbing(X_train, y_train, iterations=500, mutation_scale=0.1, seed=42):
    """Run hill climbing to optimize a linear classifier."""
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]

    # Random initial weights
    weights = rng.standard_normal(n_features) * 0.1
    bias = 0.0
    best_score = evaluate(X_train, y_train, weights, bias)

    history = [best_score]

    for i in range(iterations):
        # Mutate
        new_weights = weights + rng.standard_normal(n_features) * mutation_scale
        new_bias = bias + rng.standard_normal() * mutation_scale

        score = evaluate(X_train, y_train, new_weights, new_bias)

        if score > best_score:
            weights, bias, best_score = new_weights, new_bias, score

        history.append(best_score)

        if (i + 1) % 100 == 0:
            print(f"  Iteration {i+1:>5d} | Best F1: {best_score:.4f}")

    return weights, bias, history


def main():
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    loader.summary(X_train, X_test, y_train, y_test)

    print("\n--- HILL CLIMBING ---")
    weights, bias, history = hill_climbing(X_train, y_train, iterations=1000)

    y_pred = predict(X_test, weights, bias)
    print(f"\n--- TEST RESULTS ---")
    print(classification_report(y_test, y_pred, target_names=["Other", "Planet"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"F1 Score (test): {f1_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
