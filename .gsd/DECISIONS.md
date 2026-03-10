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
