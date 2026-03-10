---
phase: 5
plan: 1
wave: 1
---

# Plan 5.1: NEAT — NeuroEvolution of Augmenting Topologies

## Objective
Implémenter l'algorithme NEAT en utilisant la librairie `neat-python`. Ce modèle va faire évoluer non seulement les poids mais aussi la structure du réseau de neurones. Une visualisation Pygame en temps réel affichera la topologie du meilleur réseau à chaque génération.

## Context
- .gsd/SPEC.md
- shared/ (config, data_loader, metrics, resource_monitor)

## Tasks

<task type="auto">
  <name>Créer algorithms/04_neat/config-neat.txt</name>
  <files>algorithms/04_neat/config-neat.txt</files>
  <action>
    Créer le fichier de configuration NEAT avec les paramètres appropriés :
    - [NEAT] : pop_size=50, fitness_criterion=max, fitness_threshold=0.95
    - [DefaultGenome] : 
      * num_inputs=12, num_outputs=1, num_hidden=0
      * activation_options=sigmoid, node_add_prob=0.2, conn_add_prob=0.5
      * weight_mutate_rate=0.8, weight_replace_rate=0.1
    - [DefaultSpeciesSet] : compatibility_threshold=3.0
    - [DefaultStagnation] : max_stagnation=15
    - [DefaultReproduction] : survival_threshold=0.2
  </action>
  <verify>python -c "import neat; neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'algorithms/04_neat/config-neat.txt')"</verify>
  <done>Fichier de configuration valide et compatible avec neat-python.</done>
</task>

<task type="auto">
  <name>Implémenter algorithms/04_neat/main.py</name>
  <files>algorithms/04_neat/main.py</files>
  <action>
    Créer le script principal pour NEAT :
    
    1. Chargement des données.
    
    2. Logique NEAT :
       - Fonction fitness : évaluer un génome sur le train set.
       - Créer la population à partir de `config-neat.txt`.
       - Ajouter un `std_out` reporter pour le logging.
    
    3. Visualisation Pygame :
       - Dessiner le graphe du génome (cercles pour les nodes, lignes pour les connections).
       - Code couleur : Entrées (Bleu), Sorties (Rouge), Cachés (Gris).
       - Épaisseur/couleur des lignes proportionnelles aux poids.
       - Statistiques : Génération, Fitness Max, Nombre d'espèces.
    
    4. Sortie :
       - Meilleur génome sauvé (pickle).
       - Évaluation finale sur test set.
       - Sauvegarde des résultats standards.
  </action>
  <verify>python algorithms/04_neat/main.py --no-gui</verify>
  <done>NEAT fonctionnel, visualisation de la topologie active, Accuracy attendue ~85-90%.</done>
</task>

## Success Criteria
- [ ] NEAT fait évoluer la structure du réseau (complexification).
- [ ] La visualisation montre clairement les connections et les nodes cachés qui apparaissent.
- [ ] Le fitness augmente au fil des générations.
- [ ] Les résultats finaux surpassent (idélament) le GA standard.
