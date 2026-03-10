# DECISIONS.md — Architecture Decision Records

## Format
| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| ADR-01 | Python comme langage principal | Standard pour le ML, librairies NEAT disponibles | 2026-03-10 |
| ADR-02 | Pygame pour visualisation temps réel | Plus interactif que matplotlib pour l'animation en direct | 2026-03-10 |
| ADR-03 | matplotlib pour résultats finaux | Standard scientifique, qualité publication | 2026-03-10 |
| ADR-04 | Module shared/ pour code commun | DRY — éviter la duplication du data loading et des métriques | 2026-03-10 |
| ADR-05 | Hill Climbing sur perceptron simple | Point intermédiaire logique entre random et GA, simple à visualiser | 2026-03-10 |
| ADR-06 | neat-python comme librairie NEAT | Implémentation Python de référence, bien documentée | 2026-03-10 |
| ADR-07 | Split 80/20 avec seed fixe | Standard ML, reproductibilité garantie | 2026-03-10 |
| ADR-08 | StandardScaler pour normalisation | Gère mieux les outliers présents dans les données Kepler (koi_insol, koi_prad extrêmes) | 2026-03-10 |
| ADR-09 | Numpy arrays comme format de données | Plus rapide pour le calcul numérique, noms de colonnes accessibles séparément | 2026-03-10 |
| ADR-10 | Config en Python simple (pas YAML) | Suffisant pour ce projet, pas de complexité inutile | 2026-03-10 |
| ADR-11 | Seuil de classification configurable (défaut 0.5) | Curseur interactif dans la visualisation finale + calcul du seuil optimal via Youden's J | 2026-03-10 |
| ADR-12 | Decorator/context manager pour mesurer ressources | Mesure automatique temps et mémoire sans dupliquer la logique dans chaque algo | 2026-03-10 |
| ADR-13 | Interface Pygame Unifiée | Pivot : abandon du Web pour une interface 100% Python/Pygame plus intégrée | 2026-03-10 |

## Phase 8 Decisions: FrontEnd Dashboard

**Date:** 2026-03-10

### Technology Choice
- **Framework**: Pygame (pour la rapidité, l'intégration directe des algos et la cohérence visuelle).
- **Design**: Look "Cyber/Premium" avec des composants UI personnalisés (boutons, sliders, graphiques temps réel).
- **Logic**: Un seul point d'entrée `dashboard.py` qui importe et exécute les classes d'algorithmes.

### Approach
- **Fenêtre Unique** : Système d'onglets ou menus pour naviguer entre "Résumé" et "Algorithmes".
- **Paramétrage** : Sliders et boutons interactifs dans Pygame pour ajuster les itérations, etc.
- **Visualisation Intégrée** : Les animations Pygame de chaque phase sont refactorisées pour tourner dans le dashboard principal.

**Date:** 2026-03-10

### Scope
- Module shared/ avec : config.py, data_loader.py, metrics.py, visualization.py
- Normalisation par StandardScaler (fit sur train, transform sur train+test+unknown)
- Données retournées en numpy arrays, noms de colonnes accessibles séparément

### Approach
- Config en constantes Python simples
- Interface standard `evaluate()` pour les métriques
- Seuil configurable par algo (défaut 0.5)
- Calcul automatique du seuil optimal via Youden's J statistic sur la courbe ROC
- Curseur interactif (slider) dans la visualisation finale matplotlib pour ajuster le seuil et voir les métriques se recalculer en temps réel

### Constraints
- Le StandardScaler doit être fit uniquement sur les données d'entraînement (pas de data leakage)
- Le seed doit être appliqué avant tout split/shuffle
