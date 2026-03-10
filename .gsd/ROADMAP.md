# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0

## Must-Haves (from SPEC)
- [ ] 4 algorithmes principaux fonctionnels (Aléatoire, Hill Climbing, GA, NEAT)
- [ ] Visualisations Pygame temps réel pour chaque algo
- [ ] Métriques comparatives complètes (matplotlib)
- [ ] Prédictions exportées en CSV sur les données inconnues
- [ ] Exécution indépendante de chaque algorithme

## Phases

### Phase 1: Foundation — Infrastructure partagée
**Status**: ✅ Complete
**Objective**: Mettre en place le module `shared/` (data loading, metrics, config) et la structure du projet
**Deliverables**:
- `shared/config.py` — Configuration globale (seed, paths, split ratio)
- `shared/data_loader.py` — Chargement CSV, normalisation, split 80/20
- `shared/metrics.py` — Calcul de toutes les métriques (accuracy, precision, recall, F1, ROC, AUC, etc.)
- `shared/visualization.py` — Graphiques matplotlib comparatifs
- `requirements.txt` — Dépendances Python
- Structure de dossiers (`algorithms/`, `results/`)

### Phase 2: Baseline — Algorithme Aléatoire
**Status**: ✅ Complete
**Objective**: Implémenter le baseline aléatoire comme point de comparaison
**Deliverables**:
- `algorithms/01_random/main.py` — Génération de probabilités aléatoires
- Sauvegarde des prédictions et métriques
- Fenêtre Pygame simple (distribution des prédictions en temps réel)
**Dependencies**: Phase 1

### Phase 3: Hill Climbing
**Status**: ✅ Complete
**Objective**: Implémenter l'optimisation d'un perceptron par Hill Climbing
**Deliverables**:
- `algorithms/02_hill_climbing/main.py` — Perceptron + perturbation des poids
- Visualisation Pygame : courbe de fitness, état des poids
- Critère d'arrêt par stabilisation
**Dependencies**: Phase 1

### Phase 4: Algorithme Génétique Standard (GA)
**Status**: ✅ Complete
**Objective**: Implémenter un GA pour optimiser un réseau de neurones simple
**Deliverables**:
- `algorithms/03_genetic_algorithm/main.py` — Population, sélection, croisement, mutation
- Visualisation Pygame : évolution de la population, fitness par génération
- Critère d'arrêt par stabilisation
**Dependencies**: Phase 1

### Phase 5: NeuroEvolution (NEAT)
**Status**: ✅ Complete
**Objective**: Implémenter NEAT avec évolution de topologie et poids, visualisation du réseau en temps réel
**Deliverables**:
- `algorithms/04_neat/main.py` — Configuration et exécution NEAT
- `algorithms/04_neat/config-neat.txt` — Fichier de configuration NEAT
- Visualisation Pygame : réseau de neurones du meilleur individu en direct
**Dependencies**: Phase 1

### Phase 6: Résultats et Comparaison
**Status**: ✅ Complete
**Objective**: Synthèse des performances et graphiques finaux
comparatifs et les prédictions sur les données inconnues
**Deliverables**:
- Script de comparaison globale
- Graphiques matplotlib : barplots, ROC curves, matrices de confusion
- Export CSV des prédictions sur unknown_data
- Rapport de comparaison des ressources (temps, mémoire)
**Dependencies**: Phases 2-5

### Phase 7: Extensions optionnelles (HyperNEAT, WANN)
**Status**: ⬜ Not Started
**Objective**: Implémenter des algorithmes avancés si le temps le permet
**Deliverables**:
- `algorithms/05_hyperneat/main.py` — HyperNEAT
- `algorithms/06_wann/main.py` — WANN
- Intégration dans les résultats comparatifs
**Dependencies**: Phase 5

### Phase 8: Dashboard Global (Pygame)
**Status**: ⬜ Not Started
**Objective**: Créer une interface unifiée en Pygame pour piloter l'ensemble du projet, régler les hyperparamètres et visualiser l'évolution de chaque IA dans une fenêtre unique.
**Deliverables**:
- `dashboard.py` (Point d'entrée unique)
- UI Components Pygame (Boutons, Sliders, Tabs)
- Intégration des visualisations d'entraînement
- Affichage des graphiques de synthèse
**Dependencies**: Phase 6
