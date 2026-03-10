---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Algorithme Aléatoire et Visualisation Pygame

## Objective
Implémenter l'algorithme "Aléatoire" qui servira de baseline. Il doit générer des prédictions purement aléatoires, calculer les métriques, sauvegarder les résultats et afficher une animation Pygame simple montrant l'activité de l'IA (distribution des probabilités).

## Context
- .gsd/SPEC.md
- shared/ (config, data_loader, metrics, resource_monitor)

## Tasks

<task type="auto">
  <name>Implémenter algorithms/01_random/main.py</name>
  <files>algorithms/01_random/main.py</files>
  <action>
    Créer le script principal de l'algorithme aléatoire :
    
    1. Chargement des données via `shared.data_loader.load_and_prepare_data()`.
    
    2. Logique de l'algorithme :
       - Pour chaque objet (test et unknown), générer une probabilité aléatoire uniforme entre 0.0 et 1.0.
       - Utiliser `np.random.uniform` avec le SEED de la config.
    
    3. Utilisation du `ResourceMonitor` pour mesurer l'entraînement (même si quasi-instantané ici).
    
    4. Évaluation et Sauvegarde :
       - Calculer les métriques avec `shared.metrics.compute_all_metrics()`.
       - Trouver le seuil optimal avec `shared.metrics.find_optimal_threshold()`.
       - Sauvegarder métriques, ressources et prédictions (unknown) dans `results/`.
    
    5. Intégration Pygame :
       - Créer une fenêtre de 800x600.
       - Afficher en temps réel les probabilités générées sous forme d'histogramme ou de "pluie" de points.
       - Afficher les métriques actuelles (Accuracy, F1) sur l'écran Pygame.
       - Prévoir une boucle d'animation qui traite les données progressivement pour simuler un "apprentissage".
  </action>
  <verify>python algorithms/01_random/main.py --no-gui # Ajouter un flag --no-gui pour tester sans ouvrir de fenêtre</verify>
  <done>Script main.py fonctionnel, résultats sauvegardés dans results/, fenêtre Pygame s'ouvre et s'anime</done>
</task>

## Success Criteria
- [ ] L'algorithme génère des probabilités 0-1 aléatoires pour les sets test et unknown
- [ ] Les fichiers JSON de métriques et ressources sont générés dans `results/`
- [ ] Les prédictions sur `unknown_data` sont exportées en CSV
- [ ] L'animation Pygame montre visuellement la génération des probabilités
- [ ] Le script est indépendant (utilisable via `python algorithms/01_random/main.py`)
