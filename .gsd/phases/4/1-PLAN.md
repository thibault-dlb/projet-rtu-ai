---
phase: 4
plan: 1
wave: 1
---

# Plan 4.1: Algorithme Génétique Standard (GA)

## Objective
Implémenter un algorithme génétique pour optimiser les poids d'un perceptron (ou mini réseau). Utiliser une population d'individus, la sélection, le croisement et la mutation pour faire évoluer les solutions. Visualiser la diversité et la performance de la population dans Pygame.

## Context
- .gsd/SPEC.md
- shared/ (config, data_loader, metrics, resource_monitor)
- algorithms/02_hill_climbing/main.py (pour réutiliser la fonction de prédiction)

## Tasks

<task type="auto">
  <name>Implémenter algorithms/03_genetic_algorithm/main.py</name>
  <files>algorithms/03_genetic_algorithm/main.py</files>
  <action>
    Créer le script principal pour le GA :
    
    1. Chargement des données.
    
    2. Logique GA :
       - Population de ~100 individus.
       - Gène = Vecteur de 13 floats (12 weights + 1 bias).
       - Évaluation de Fitness = Accuracy sur train set.
       - Boucle Évolutionnaire :
         * Sélection : Tournoi (taille 3).
         * Croisement : Uniform Crossover.
         * Mutation : Gaussian Noise sur certains gènes.
         * Élitisme : Conserver le meilleur individu.
    
    3. Visualisation Pygame :
       - Graphique montrant le fitness Max, Moyen et Min par génération.
       - Visualisation de la "dispersion" de la population (ex: plot 2D des deux premiers poids).
       - Barre de progression des générations.
    
    4. Critère d'arrêt :
       - Nombre de générations fixe (ex: 100) ou stabilisation.
    
    5. Sortie :
       - Évaluation finale du meilleur individu sur test set.
       - Sauvegarde des résultats standards.
  </action>
  <verify>python algorithms/03_genetic_algorithm/main.py --no-gui</verify>
  <done>GA fonctionnel, Accuracy >= 83% (devrait être au moins aussi bon que Hill Climbing), résultats exportés.</done>
</task>

## Success Criteria
- [ ] La population évolue et le fitness moyen augmente.
- [ ] Le croisement et la mutation sont correctement implémentés.
- [ ] La visualisation Pygame permet de voir la convergence de la population.
- [ ] Performance finale compétitive avec le Hill Climbing.
