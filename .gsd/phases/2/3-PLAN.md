---
phase: 2
plan: 3
wave: 2
---

# Plan 2.3: Unified AI Engine & Metrics

## Objective
Créer une interface unifiée pour tous les algorithmes (Random, Hill Climbing, GA, NEAT) et un système de métriques en temps réel (matrice de confusion, F1, seuil de décision variable). Ce module sera consommé directement par l'interface Dear PyGui en Phase 3.

## Context
- .gsd/SPEC.md (REQ-08: matrice de confusion et F1 en temps réel)
- data_loader.py
- baseline_random.py, baseline_hill_climbing.py
- genetic_algorithm.py (Plan 2.1)
- neat_engine.py (Plan 2.2)

## Tasks

<task type="auto">
  <name>Create unified AI engine interface</name>
  <files>ai_engine.py</files>
  <action>
    Create ai_engine.py with:

    1. `AlgorithmType` enum: RANDOM, HILL_CLIMBING, GA, NEAT

    2. `TrainingResult` dataclass:
       - best_fitness: float
       - test_f1: float
       - confusion_matrix: np.ndarray
       - history: dict (per-generation/iteration metrics)
       - predictions: np.ndarray
       - topology_info: dict (for NEAT: nodes, connections; for others: weights shape)
       - training_time: float

    3. `AIEngine` class:
       - `__init__(algorithm: AlgorithmType, **params)`
       - `train(X_train, y_train, X_test, y_test, callback=None)` -> TrainingResult
           * callback(iteration, metrics_dict) — called each generation/iteration for live updates
           * metrics_dict contains: best_fitness, avg_fitness, iteration, elapsed_time
       - `predict(X, threshold=0.5)` -> np.ndarray
       - `compute_metrics(y_true, y_pred)` -> dict with precision, recall, f1, confusion_matrix
       - `adjust_threshold(threshold)` — recalculate predictions with a new decision threshold

    Important: The callback mechanism is critical — it's how the GUI will receive live updates.
    Important: The threshold adjustment must NOT retrain — just re-threshold existing predictions.
    Important: Each algorithm type delegates to the actual engine (hill_climbing, GA, NEAT).
  </action>
  <verify>python -c "from ai_engine import AIEngine, AlgorithmType; print('Import OK')"</verify>
  <done>ai_engine.py imports without error and provides a unified AlgorithmType/AIEngine interface</done>
</task>

<task type="auto">
  <name>Validate unified engine with all algorithms</name>
  <files>ai_engine.py</files>
  <action>
    Add `if __name__ == "__main__"` block that:
    1. Loads data
    2. Runs each algorithm type (RANDOM, HILL_CLIMBING, GA, NEAT) with small parameters
    3. Prints a comparison table: Algorithm | F1 | Time | Topology
    4. Tests threshold adjustment: for threshold 0.3, 0.5, 0.7 show F1 change

    This validates that all algorithms work through the unified interface.
  </action>
  <verify>python ai_engine.py</verify>
  <done>All 4 algorithms run via AIEngine, comparison table printed, threshold adjustment works</done>
</task>

## Success Criteria
- [ ] All 4 algorithms (Random, Hill Climbing, GA, NEAT) are accessible through AIEngine
- [ ] Callback mechanism works (prints live metrics during training)
- [ ] Threshold adjustment changes predictions without retraining
- [ ] Comparison table shows reasonable relative performance
