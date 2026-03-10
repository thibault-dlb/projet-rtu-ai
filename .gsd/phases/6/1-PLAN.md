---
phase: 6
plan: 1
wave: 1
---

# Plan 6.1: Résultats comparatifs et prédictions finales

## Objective
Générer l'ensemble des graphiques comparatifs pour analyser les performances des différents algorithmes (Aléatoire, Hill Climbing, GA, NEAT). Produire également un rapport de synthèse et s'assurer que les prédictions sur les données inconnues sont complètes.

## Context
- .gsd/SPEC.md
- results/metrics/ (*_metrics.json, *_resources.json)
- shared/visualization.py

## Tasks

<task type="auto">
  <name>Créer algorithms/compare_results.py</name>
  <files>algorithms/compare_results.py</files>
  <action>
    Créer un script de synthèse qui :
    
    1. Importe `shared.visualization`.
    2. Appelle `generate_all_plots()` pour créer :
       - `metrics_comparison.png`
       - `roc_curves.png` (à implémenter si pas encore fait)
       - `resource_comparison.png`
    3. Génère un tableau markdown récapitulatif dans la console et dans `results/summary.md`.
    4. Identifie le meilleur algorithme basé sur le F1-Score.
  </action>
  <verify>python algorithms/compare_results.py</verify>
  <done>Graphiques générés dans results/plots/, fichier summary.md créé.</done>
</task>

<task type="auto">
  <name>Finaliser shared/visualization.py (ROC Curves)</name>
  <files>shared/visualization.py</files>
  <action>
    S'assurer que `plot_roc_curves` et `plot_resource_comparison` sont bien implémentés et fonctionnels.
    Vérifier que le style des graphiques est premium et prêt pour une présentation.
  </action>
  <verify>python -c "import shared.visualization as v; v.generate_all_plots()"</verify>
  <done>Tous les types de graphiques (ROC, Ressources, Métriques) sont fonctionnels.</done>
</task>

## Success Criteria
- [ ] Tous les algorithmes (Phase 2 à 5) sont comparés sur un même graphique.
- [ ] Un classement clair des performances est établi.
- [ ] Les graphiques sont visuellement réussis (premium).
- [ ] Le projet est prêt pour une revue finale.
