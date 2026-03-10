# STATE.md — Project Memory

> **Last Updated**: 2026-03-10
> **Current Phase**: 2 (completed)
> **Session**: 1

## Project Summary
Classification d'exoplanètes Kepler avec une progression d'algorithmes :
Aléatoire → Hill Climbing → GA → NEAT → (HyperNEAT, WANN)

## Key Decisions
- Python 3.x, Pygame pour temps réel, matplotlib pour résultats
- Module shared/ pour code commun (DRY), exécution indépendante par algo
- Données Kepler : 12 features, ~7326 known, ~1877 unknown
- Split 80/20 avec seed fixe, StandardScaler (fit sur train uniquement)
- Seuil configurable (défaut 0.5) + slider interactif + Youden's J optimal

## Current Position
- [x] SPEC.md finalized
- [x] ROADMAP.md created
- [x] Phase 1: Foundation (verified)
- [x] Phase 2: Random Baseline (verified)
- [ ] Phase 3: Hill Climbing
- [ ] Phase 4: GA
- [ ] Phase 5: NEAT

## Last Session Summary
Phase 2 terminée.
- Algorithme Aléatoire implémenté dans `algorithms/01_random/main.py`.
- Visualisation Pygame (animation de distribution) opérationnelle.
- Baseline établie : Accuracy ~50%, AUC ~0.5.
- Résultats sauvegardés dans `results/`.

## Next Steps
1. /plan 3 (Hill Climbing)
