---
phase: 8
plan: 1
wave: 1
---

# Plan 8.1: Dashboard Global — Pygame Launcher

## Objective
Créer un centre de contrôle unique en Pygame pour piloter tous les algorithmes d'IA (Random, Hill Climbing, GA, NEAT). L'interface doit permettre de régler les paramètres, de lancer l'entraînement/test, et de visualiser les progrès en temps réel sans quitter l'application.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (Pivot Pygame)
- shared/config.py

## Tasks

<task type="auto">
  <name>Créer le Framework UI Pygame (shared/ui.py)</name>
  <files>shared/ui.py</files>
  <action>
    Développer des composants réutilisables en Pygame pour le dashboard :
    1. `Button` : Avec états hover, press et callback.
    2. `Slider` : Pour régler les hyperparamètres (itérations, pop size).
    3. `Tabs` : Pour naviguer entre l'Overview et les différentes IAs.
    4. `ProgressBar` : Pour le suivi de progression.
  </action>
  <verify>Script de test rapide lançant une fenêtre avec ces composants.</verify>
  <done>Composants UI fonctionnels et esthétiques.</done>
</task>

<task type="auto">
  <name>Refactoriser les Visualiseurs d'Algos</name>
  <files>algorithms/*/main.py</files>
  <action>
    Modifier les scripts `main.py` pour qu'ils puissent être importés comme classes/fonctions et s'injecter dans une surface Pygame parente au lieu de créer leur propre fenêtre `pygame.display.set_mode`.
  </action>
  <verify>Vérifier que les algos peuvent tourner "en silence" ou dans une surface donnée.</verify>
  <done>Algos isolés de leur gestion de fenêtre display.</done>
</task>

<task type="auto">
  <name>Développer dashboard.py</name>
  <files>dashboard.py</files>
  <action>
    Assembler le tout dans `dashboard.py` :
    1. Menu latéral pour choisir l'algo.
    2. Panneau central de contrôle (Hyperparamètres + Bouton Start/Stop).
    3. Zone de visualisation (Graphiques temps réel + logs simplifiés).
    4. Intégration du résumé global des performances.
  </action>
  <verify>Lancement de `python dashboard.py`.</verify>
  <done>Interface unifiée fonctionnelle.</done>
</task>

## Success Criteria
- [ ] Un seul script `dashboard.py` contrôle tout le projet.
- [ ] Les paramètres sont réglables via des sliders/boutons (pas de ligne de commande).
- [ ] Le look est "Premium" (Cyber-style, animations fluides).
- [ ] Les visualisations d'entraînement s'affichent correctement dans l'interface principale.
