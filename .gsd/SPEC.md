# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
Développer une suite logicielle native pour Windows permettant la classification d'objets célestes (exoplanètes) à l'aide d'algorithmes évolutionnaires (NEAT) et d'IA classiques, pilotée par une interface graphique performante (Dear PyGui) offrant des visualisations 3D en temps réel et des rapports vidéo automatisés via Manim.

## Goals
1. **Performance Maximale** : Atteindre 60 FPS sur l'interface graphique grâce à l'accélération matérielle (DirectX/OpenGL).
2. **IA Évolutionnaire** : Implémenter NEAT et des approches par recherche (Hill Climbing, GA) avec support GPU (PyTorch/CUDA).
3. **Analyse Étendue** : Réaliser des Grid Search massifs (40 000+ combinaisons) visualisés en 3D (Population vs Générations).
4. **Reporting Visuel** : Générer des vidéos Manim illustrant l'évolution topologique des réseaux de neurones.
5. **Fluidité Système** : Utiliser le multi-threading et multi-processing pour garantir que l'UI ne freeze jamais durant les calculs.

## Non-Goals (Out of Scope)
- Déploiement Web ou Mobile (Focus exclusif sur Desktop Native Windows).
- Analyse d'autres types de données astronomiques que les Kepler Objects of Interest (KOI) fournis.
- Pipeline d'ingestion de données en temps réel depuis les serveurs de la NASA (Utilisation de datasets locaux).

## Users
- **Ingénieurs en IA/Data Scientists** : Pour expérimenter sur les topologies de réseaux de neurones pour la classification.
- **Passionnés d'Astronomie** : Pour visualiser et comprendre comment une IA apprend à identifier des planètes.

## Constraints
- **Matérielles** : Optimisé pour Ryzen 5 5500 (12 threads) et RTX 2070 Super (8GB VRAM).
- **Logicielles** : Python 3.x, Windows, Dear PyGui, PyTorch (CUDA), neat-python, Manim.
- **Interface** : Esthétique "Engineering/Gaming" (style Opera GX).

## Success Criteria
- [ ] Classification avec un score F1 supérieur à 0.85 sur le dataset `known_data`.
- [ ] Interface fluide à 60 FPS pendant l'entraînement massif.
- [ ] Visualisation 3D en direct du Grid Search sans ralentissement majeur du pipeline de calcul.
- [ ] Export réussi d'une vidéo Manim montrant l'évolution du meilleur génome.
