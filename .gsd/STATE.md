# STATE.md — Project Memory

> **Last Updated**: 2026-03-10
> **Current Phase**: 6 (completed)
> **Session**: 1

## Project Summary
Classification d'exoplanètes Kepler avec une progression d'algorithmes :
Aléatoire → Hill Climbing → GA → NEAT

## Key Decisions
- Python 3.x, Pygame pour temps réel, matplotlib pour résultats
- Module shared/ pour code commun (DRY), exécution indépendante par algo
- Données Kepler : 12 features, ~7326 known, ~1877 unknown
- Split 80/20 avec seed fixe, StandardScaler (fit sur train uniquement)
- Seuil configurable (défaut 0.5) + slider interactif + Youden's J optimal
- **Paramétrabilité** : Itérations (`--iterations`) et Générations/Pop (`--generations`, `--pop-size`) réglables via CLI pour chaque modèle.

## Constraints & Notes
- **Low Compute**: Les tests actuels sont effectués avec peu d'itérations/générations pour validation structurelle uniquement. Les "vrais" entraînements seront lancés ultérieurement sur une machine plus puissante.

## Current Position
- [x] Phase 1: Foundation (Verified)
- [x] Phase 2: Random Baseline (Verified)
- [x] Phase 3: Hill Climbing (Verified)
- [x] Phase 4: GA (Verified)
- [x] Phase 5: NEAT (Verified)
- [x] Phase 6: Results Comparison (Verified)
- [ ] Phase 7: Extensions (Optional/Future)
- [ ] Phase 8: FrontEnd Dashboard

## Final Results Overview
| Algorithme           | Accuracy | F1-Score | ROC AUC |
|:---------------------|:---------|:---------|:--------|
| Hill Climbing        | 0.8315   | 0.7883   | 0.8942  |
| Algorithme Génétique | 0.8315   | 0.7777   | 0.8848  |
| NEAT (20 gens)       | 0.7838   | 0.7217   | 0.8315  |
| Aléatoire            | 0.5082   | 0.4415   | 0.5073  |

## Next Steps
1. Projet prêt pour exécution intensive.
2. Facultatif : Implémenter HyperNEAT ou WANN.
