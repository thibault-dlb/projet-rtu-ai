---
phase: 1
plan: 3
wave: 2
---

# Plan 1.3: Visualisation comparative et Requirements

## Objective
Créer le module `visualization.py` pour les graphiques comparatifs finaux (matplotlib) avec le slider interactif de seuil, et le fichier `requirements.txt` avec toutes les dépendances.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-11 — slider interactif)
- shared/config.py (créé dans Plan 1.1)
- shared/metrics.py (créé dans Plan 1.2)

## Tasks

<task type="auto">
  <name>Créer shared/visualization.py</name>
  <files>shared/visualization.py</files>
  <action>
    Implémenter le module de visualisation comparative finale (matplotlib) :
    
    1. Fonction `plot_metrics_comparison(results_dir)` :
       - Charger tous les fichiers *_metrics.json depuis results/metrics/
       - Créer un barplot groupé comparant accuracy, precision, recall, F1 pour chaque algo
       - Sauvegarder dans results/plots/metrics_comparison.png
    
    2. Fonction `plot_confusion_matrices(results_dir)` :
       - Charger les matrices de confusion de chaque algo
       - Afficher en subplot grid (une matrice par algo)
       - Sauvegarder dans results/plots/confusion_matrices.png
    
    3. Fonction `plot_roc_curves(results_dir)` :
       - Superposer les courbes ROC de tous les algos sur un même graphique
       - Afficher AUC dans la légende
       - Sauvegarder dans results/plots/roc_curves.png
    
    4. Fonction `plot_resource_comparison(results_dir)` :
       - Barplots comparant temps d'entraînement et mémoire utilisée par algo
       - Sauvegarder dans results/plots/resource_comparison.png
    
    5. Fonction `plot_prediction_distributions(results_dir)` :
       - Histogrammes des probabilités prédites sur unknown_data par algo
       - Sauvegarder dans results/plots/prediction_distributions.png
    
    6. Fonction `interactive_threshold_viewer(results_dir)` :
       - Fenêtre matplotlib interactive avec slider (matplotlib.widgets.Slider)
       - Le slider ajuste le seuil de 0.0 à 1.0
       - En temps réel : recalculer et afficher accuracy, precision, recall, F1
       - Afficher aussi la matrice de confusion qui se met à jour
       - Afficher le seuil optimal Youden's J comme ligne verticale de référence
       - Un dropdown ou des boutons pour sélectionner l'algo à examiner
    
    7. Fonction `generate_all_plots(results_dir)` :
       - Appelle toutes les fonctions ci-dessus (sauf interactive)
       - Affiche un résumé des graphiques générés
    
    IMPORTANT :
    - Utiliser un style matplotlib propre (plt.style.use('seaborn-v0_8-darkgrid') ou similaire)
    - Couleurs distinctes et cohérentes pour chaque algo
    - Labels et titres en français
    - Les fonctions doivent gérer le cas où certains algos n'ont pas encore de résultats
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); from shared.visualization import generate_all_plots, interactive_threshold_viewer; print('visualization module importable OK')"</verify>
  <done>Module importable, toutes les fonctions définies et documentées</done>
</task>

<task type="auto">
  <name>Créer requirements.txt</name>
  <files>requirements.txt</files>
  <action>
    Créer le fichier requirements.txt avec toutes les dépendances Python du projet :
    
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - pygame
    - neat-python
    - tracemalloc (stdlib, pas besoin de le lister)
    
    Ne PAS fixer de versions spécifiques sauf si nécessaire pour compatibilité.
    Ajouter un commentaire en-tête expliquant le projet.
  </action>
  <verify>python -c "print('requirements.txt exists')" && type requirements.txt</verify>
  <done>requirements.txt contient toutes les dépendances nécessaires</done>
</task>

## Success Criteria
- [ ] `visualization.py` contient toutes les fonctions de visualisation comparative
- [ ] `interactive_threshold_viewer()` implémente le slider interactif avec métriques dynamiques
- [ ] Les graphiques sont sauvegardés en PNG dans results/plots/
- [ ] `requirements.txt` liste toutes les dépendances
- [ ] Style visuel propre et cohérent (labels français, couleurs distinctes)
