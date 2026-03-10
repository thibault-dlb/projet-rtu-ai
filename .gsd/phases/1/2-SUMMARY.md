# Plan 1.2 Summary: Métriques et Mesure de Ressources

## Objective
Mettre en place les outils d'évaluation de performance et de suivi de consommation de ressources.

## Changes
- Créé `shared/metrics.py` : calcul exhaustif des métriques (Accuracy, Precision, Recall, F1, ROC AUC, Log-loss, Matrice de confusion) et découverte du seuil optimal via Youden's J statistic.
- Créé `shared/resource_monitor.py` : context manager utilisant `tracemalloc` et `time.perf_counter` pour mesurer le temps et la mémoire RAM pic.

## Verification
- Script `verify_plan_1_2.py` exécuté.
- Résultats :
  - Métriques validées sur données de test (Accuracy 1.0, AUC 1.0).
  - Youden's J validé (Seuil optimal 0.70).
  - Monitor validé (Temps et Mémoire mesurés correctement en dehors du bloc contextuel).

## Verdict
**PASS**
