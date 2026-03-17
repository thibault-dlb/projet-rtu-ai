---
phase: 2
plan: 2
wave: 1
---

# Plan 2.2: NEAT Integration

## Objective
Intégrer `neat-python` pour l'évolution topologique des réseaux de neurones. NEAT est le cœur du projet — il fait évoluer la structure même du réseau (ajout de nœuds, de connexions) en plus des poids.

## Context
- .gsd/SPEC.md
- data_loader.py (ExoplanetDataLoader, FEATURES)
- neat-python documentation (https://neat-python.readthedocs.io/)

## Tasks

<task type="auto">
  <name>Create NEAT configuration file</name>
  <files>neat_config.ini</files>
  <action>
    Create neat_config.ini with:

    [NEAT]
    - fitness_criterion = max
    - fitness_threshold = 0.95
    - pop_size = 100
    - reset_on_extinction = True

    [DefaultGenome]
    - num_inputs = 12 (our 12 features)
    - num_outputs = 1 (binary classification: sigmoid output)
    - num_hidden = 0 (start minimal, let NEAT add nodes)
    - activation_default = sigmoid
    - activation_mutate_rate = 0.0 (keep sigmoid for classification)
    - feed_forward = True
    - Initial connection = full_nodirect
    - Reasonable mutation rates for weights, bias, add_node, add_connection

    [DefaultSpeciesSet]
    - compatibility_threshold = 3.0

    [DefaultStagnation]
    - species_fitness_func = max
    - max_stagnation = 15
    - species_elitism = 2

    [DefaultReproduction]
    - elitism = 2
    - survival_threshold = 0.2
  </action>
  <verify>python -c "import neat; config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat_config.ini'); print('Config OK')"</verify>
  <done>neat_config.ini loads without error via neat.Config</done>
</task>

<task type="auto">
  <name>Implement NEAT training engine</name>
  <files>neat_engine.py</files>
  <action>
    Create neat_engine.py with:

    1. `NeatEngine` class:
       - `__init__(config_path='neat_config.ini')`
       - `eval_genomes(genomes, config)` — fitness function:
           * Create FeedForwardNetwork from genome
           * Run all training samples through the network
           * Compute F1 score as fitness
           * Store fitness on genome.fitness
       - `run(X_train, y_train, generations=100)` — run NEAT evolution
           * Uses neat.Population
           * Adds StdOutReporter and StatisticsReporter
           * Returns: best_genome, stats (for later visualization)
       - `predict(genome, config, X)` — predict with a specific genome
       - `evaluate_best(X_test, y_test)` — evaluate the best genome on test data

    2. History tracking:
       - Store per-generation: best_fitness, avg_fitness, num_species, best_genome_topology (num_nodes, num_connections)
       - Return as dict for future visualization

    Important: neat-python's eval_genomes receives (genomes, config) where genomes is a list of (genome_id, genome) tuples.
    Important: Use neat.nn.FeedForwardNetwork.create(genome, config) to build networks.
    Important: Network output is a single float — threshold at 0.5 for classification.
  </action>
  <verify>python -c "from neat_engine import NeatEngine; print('Import OK')"</verify>
  <done>neat_engine.py imports without error and contains NeatEngine class</done>
</task>

<task type="auto">
  <name>Run NEAT training and validate performance</name>
  <files>neat_engine.py</files>
  <action>
    Add `if __name__ == "__main__"` block:
    1. Load data with ExoplanetDataLoader
    2. Run NeatEngine with 50 generations
    3. Evaluate best genome on test set
    4. Print: classification_report, confusion_matrix, F1 score
    5. Print topology info: number of nodes and connections in best genome

    Expected: F1 should be >= 0.80 (better than GA/Hill Climbing thanks to topology evolution).
  </action>
  <verify>python neat_engine.py</verify>
  <done>NEAT runs to completion, achieves F1 >= 0.75 on test set, topology info is printed</done>
</task>

## Success Criteria
- [ ] `neat_config.ini` is valid and loads correctly
- [ ] `neat_engine.py` runs complete NEAT evolution without errors
- [ ] Best genome achieves F1 >= 0.75 on test set
- [ ] Training history includes topology metrics (nodes, connections per generation)
