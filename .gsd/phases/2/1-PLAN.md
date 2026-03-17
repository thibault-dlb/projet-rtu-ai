---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Genetic Algorithm Engine

## Objective
Créer le moteur d'algorithme génétique (GA) pour la classification binaire. C'est le pont entre le Hill Climbing (un seul individu) et NEAT (topologies variables). Le GA utilise une population de perceptrons avec sélection, crossover et mutation.

## Context
- .gsd/SPEC.md
- data_loader.py (ExoplanetDataLoader, FEATURES, TARGET)
- baseline_hill_climbing.py (sigmoid, predict, evaluate patterns)

## Tasks

<task type="auto">
  <name>Implement Genetic Algorithm engine</name>
  <files>genetic_algorithm.py</files>
  <action>
    Create genetic_algorithm.py with:

    1. `Individual` class:
       - `weights` (np.ndarray of shape [12]), `bias` (float), `fitness` (float)
       - `predict(X)` — sigmoid perceptron, same as hill climbing
       - `evaluate(X, y)` — compute and store F1 score as fitness

    2. `GeneticAlgorithm` class:
       - `__init__(pop_size, n_features=12, mutation_rate=0.1, mutation_scale=0.2, crossover_rate=0.7, elite_ratio=0.1, seed=42)`
       - `initialize()` — create random population
       - `evaluate_population(X, y)` — compute fitness for all individuals
       - `select_parents()` — tournament selection (tournament size = 3)
       - `crossover(parent1, parent2)` — uniform crossover on weights + bias
       - `mutate(individual)` — gaussian noise mutation
       - `evolve()` — one generation: select, crossover, mutate, elitism
       - `run(X_train, y_train, generations)` — full training loop, returns history dict with keys: 'best_fitness', 'avg_fitness', 'best_individual'
       - Log best/avg fitness per generation

    Important: Do NOT use PyTorch here. This is pure NumPy to keep GA simple and fast for CPU.
    Important: Use the same sigmoid/predict pattern from baseline_hill_climbing.py for consistency.
  </action>
  <verify>python -c "from genetic_algorithm import GeneticAlgorithm; print('Import OK')"</verify>
  <done>genetic_algorithm.py imports without error and contains GeneticAlgorithm class</done>
</task>

<task type="auto">
  <name>Test GA with training run</name>
  <files>genetic_algorithm.py</files>
  <action>
    Add a `if __name__ == "__main__"` block that:
    1. Loads data with ExoplanetDataLoader
    2. Runs GA with pop_size=50, generations=100
    3. Evaluates best individual on test set
    4. Prints classification_report, confusion_matrix, F1 score
    5. Prints convergence info (best fitness at gen 1, 50, 100)

    Expected: F1 should be >= Hill Climbing baseline (0.75) given population search.
  </action>
  <verify>python genetic_algorithm.py</verify>
  <done>GA runs to completion, prints F1 >= 0.70 on test set</done>
</task>

## Success Criteria
- [ ] `genetic_algorithm.py` runs end-to-end without errors
- [ ] GA achieves F1 >= 0.70 on test set (competitive with Hill Climbing)
- [ ] Training history (best/avg fitness per generation) is returned for future visualization
