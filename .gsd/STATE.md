# STATE.md — Project Memory

> **Last Updated**: 2026-03-10
> **Current Phase**: 3 (completed)
> **Session**: 1

## Project Summary
Classification d'exoplanètes Kepler avec une progression d'algorithmes :
Aléatoire → Hill Climbing → GA → NEAT → (HyperNEAT, WANN)

## Key Decisions
- Python 3.x, Pygame pour temps réel, matplotlib pour résultats
- Module shared/ pour code commun, exécution indépendante par algo
- Données Kepler : 12 features, ~7326 known, ~1877 unknown
- Split 80/20 avec seed fixe, StandardScaler (fit sur train uniquement)
- Seuil configurable (défaut 0.5) + slider interactif + Youden's J optimal

## Current Position
- [x] SPEC.md finalized
- [x] ROADMAP.md created
- [x] Phase 1: Foundation (verified)
- [x] Phase 2: Random Baseline (verified)
- [x] Phase 3: Hill Climbing (verified)
- [ ] Phase 4: GA
- [ ] Phase 5: NEAT

## Last Session Summary
Phase 3 terminée.
- Hill Climbing sur perceptron implémenté.
- Accuracy Test : 83.15% (Progression majeure).
- Visualisation Pygame interactive (poids + fitness).

## Next Steps
1. /plan 4 (Genetic Algorithm)
