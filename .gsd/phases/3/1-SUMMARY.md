# Plan 3.1 Summary: Hill Climbing

## Objective
Optimiser un perceptron simple via perturbations aléatoires des poids (Hill Climbing).

## Changes
- Créé `algorithms/02_hill_climbing/main.py`.
- Architecture : Perceptron (12 features + bias + Sigmoid).
- Algorithme : Hill Climbing avec critère de stabilisation.
- Visualisation : Courbe de fitness et barres de poids dynamiques dans Pygame.

## Verification
- Convergence atteinte à l'itération ~1500.
- Accuracy Test : **83.15%** (vs ~51% Random).
- ROC AUC : **0.8942**.
- Fichiers de résultats générés et validés.

## Verdict
**PASS**
