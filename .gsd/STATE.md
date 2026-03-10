# STATE.md — Project Memory

> **Last Updated**: 2026-03-10
> **Current Phase**: 1 — Planning complete
> **Session**: 1

## Project Summary
Classification d'exoplanètes Kepler avec une progression d'algorithmes :
Aléatoire → Hill Climbing → GA → NEAT → (HyperNEAT, WANN)

## Key Decisions
- Python 3.x, Pygame pour temps réel, matplotlib pour résultats
- Module shared/ pour code commun (DRY), exécution indépendante par algo
- Données Kepler : 12 features, ~7328 known, ~1877 unknown
- Split 80/20 avec seed fixe, StandardScaler (fit sur train uniquement)
- Seuil configurable (défaut 0.5) + slider interactif + Youden's J optimal
- ResourceMonitor (context manager) pour temps et mémoire

## Current Position
- [x] SPEC.md finalized
- [x] ROADMAP.md created
- [x] Phase 1: Planned (3 plans, 2 waves)
- [ ] Phase 1: Execution
- [ ] Phase 2: Random baseline
- [ ] Phase 3: Hill Climbing
- [ ] Phase 4: GA
- [ ] Phase 5: NEAT
- [ ] Phase 6: Results
- [ ] Phase 7: Extensions

## Phase 1 Plans
- **Plan 1.1** (Wave 1): Config + Data Loader — 3 tasks
- **Plan 1.2** (Wave 2): Métriques + Resource Monitor — 2 tasks
- **Plan 1.3** (Wave 2): Visualisation + Requirements — 2 tasks

## Next Steps
1. /execute 1

## Notes
- Aucune pour le moment
