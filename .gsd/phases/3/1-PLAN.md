---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Hill Climbing — Optimisation d'un Perceptron

## Objective
Implémenter l'algorithme Hill Climbing pour optimiser les poids d'un perceptron simple (12 features + bias). L'algorithme doit itérer jusqu'à stabilisation de la performance, avec une visualisation Pygame montrant l'évolution de la fitness et l'état des poids en temps réel.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-05)
- shared/ (config, data_loader, metrics, resource_monitor)

## Tasks

<task type="auto">
  <name>Implémenter algorithms/02_hill_climbing/main.py</name>
  <files>algorithms/02_hill_climbing/main.py</files>
  <action>
    Créer le script principal pour le Hill Climbing :
    
    1. Chargement des données normalisées via `shared.data_loader.load_and_prepare_data()`.
    
    2. Architecture du Perceptron :
       - Vecteur de poids `W` (taille 12) et un biais `b`.
       - Fonction d'activation Sigmoid pour obtenir une probabilité : `y = 1 / (1 + exp(-(X.W + b)))`.
    
    3. Boucle Hill Climbing :
       - Fitness = Accuracy sur l'ensemble d'entraînement (ou négatif de la Log-Loss).
       - À chaque itération :
         * Sauvegarder l'état actuel (weights/bias/fitness).
         * Perturber légèrement les poids : `new_W = W + uniform(-eps, eps)`.
         * Calculer la nouvelle fitness.
         * Si `new_fitness > current_fitness`, valider le changement.
         * Sinon, annuler (rollback).
    
    4. Critère d'arrêt :
       - Nombre maximum d'itérations (ex: 10,000).
       - Stabilisation : arrêt si pas d'amélioration significative (ex: < 1e-5) sur les 500 dernières itérations.
    
    5. Visualisation Pygame :
       - Courbe de fitness en temps réel (Historique des scores).
       - Représentation visuelle des poids (ex: barres verticales dont la hauteur/couleur varie avec la valeur).
       - Texte affichant l'itération actuelle, la fitness et les métriques.
    
    6. Sortie :
       - Évaluation finale sur l'ensemble de TEST.
       - Sauvegarde des métriques, ressources et prédictions (set unknown).
  </action>
  <verify>python algorithms/02_hill_climbing/main.py --no-gui</verify>
  <done>Hill Climbing opérationnel, Accuracy > 0.5 (normalement ~70-80% pour ce dataset), fichiers de résultats générés.</done>
</task>

## Success Criteria
- [ ] Le Hill Climbing améliore la fitness au fil des itérations.
- [ ] La visualisation Pygame montre l'évolution de l'apprentissage.
- [ ] L'algorithme s'arrête automatiquement par stabilisation ou limite d'itérations.
- [ ] Les métriques enregistrées montrent une nette amélioration par rapport au baseline aléatoire.
- [ ] Les prédictions sur `unknown_data` sont exportées.
