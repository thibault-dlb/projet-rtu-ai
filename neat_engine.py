"""
neat_engine.py
NEAT (NeuroEvolution of Augmenting Topologies) engine for exoplanet classification.

Uses neat-python to evolve both the topology and weights of neural networks.
Tracks per-generation metrics for live visualization.
"""

import neat
import numpy as np
import time
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from data_loader import ExoplanetDataLoader


class NeatEngine:
    """Evolutionary neural network engine using NEAT."""

    def __init__(self, config_path='neat_config.ini'):
        self.config_path = config_path
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.best_genome = None
        self.stats = None
        self._X_train = None
        self._y_train = None
        self._callback = None
        self._start_time = None
        self._history = {
            'best_fitness': [],
            'avg_fitness': [],
            'num_species': [],
            'best_nodes': [],
            'best_connections': [],
        }

    # ------------------------------------------------------------------
    def _eval_genomes(self, genomes, config):
        """Fitness function called by neat-python each generation."""
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            y_pred = np.array([
                1 if net.activate(x)[0] >= 0.5 else 0
                for x in self._X_train
            ])

            genome.fitness = f1_score(self._y_train, y_pred, zero_division=0)

    # ------------------------------------------------------------------
    def _generation_reporter(self, gen, population):
        """Called after each generation to record metrics."""
        fitnesses = [g.fitness for g in population.population.values() if g.fitness is not None]

        if not fitnesses:
            return

        best_f = max(fitnesses)
        avg_f = np.mean(fitnesses)
        n_species = len(population.species.species)

        # Get best genome topology info
        best_genome = max(population.population.values(), key=lambda g: g.fitness or 0)
        n_nodes = len(best_genome.nodes)
        n_conns = len([c for c in best_genome.connections.values() if c.enabled])

        self._history['best_fitness'].append(best_f)
        self._history['avg_fitness'].append(avg_f)
        self._history['num_species'].append(n_species)
        self._history['best_nodes'].append(n_nodes)
        self._history['best_connections'].append(n_conns)

        if self._callback:
            self._callback(gen, {
                'best_fitness': best_f,
                'avg_fitness': avg_f,
                'iteration': gen,
                'num_species': n_species,
                'best_nodes': n_nodes,
                'best_connections': n_conns,
                'elapsed_time': time.time() - self._start_time,
            })

    # ------------------------------------------------------------------
    def run(self, X_train, y_train, generations=50, callback=None):
        """Run NEAT evolution. Returns (best_genome, history)."""
        self._X_train = X_train
        self._y_train = y_train
        self._callback = callback
        self._start_time = time.time()

        # Reset history
        for key in self._history:
            self._history[key] = []

        population = neat.Population(self.config)
        population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        population.add_reporter(self.stats)

        # Custom generation callback
        class _GenReporter(neat.reporting.BaseReporter):
            def __init__(self, engine, pop_ref):
                self.engine = engine
                self.pop_ref = pop_ref
                self.gen = 0

            def post_evaluate(self, config, population, species, best_genome):
                self.engine._generation_reporter(self.gen, self.pop_ref)
                self.gen += 1

        gen_reporter = _GenReporter(self, population)
        population.add_reporter(gen_reporter)

        self.best_genome = population.run(self._eval_genomes, generations)

        return self.best_genome, self._history

    # ------------------------------------------------------------------
    def predict(self, genome, X, threshold=0.5):
        """Predict using a specific genome."""
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return np.array([
            1 if net.activate(x)[0] >= threshold else 0
            for x in X
        ])

    # ------------------------------------------------------------------
    def evaluate_best(self, X_test, y_test):
        """Evaluate the best genome on test data."""
        if self.best_genome is None:
            raise ValueError("No best genome — run training first.")

        y_pred = self.predict(self.best_genome, X_test)
        return {
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, target_names=["Other", "Planet"]),
            'predictions': y_pred,
        }

    # ------------------------------------------------------------------
    def get_topology_info(self, genome=None):
        """Get topology info for a genome (default: best)."""
        if genome is None:
            genome = self.best_genome
        if genome is None:
            return {}

        n_nodes = len(genome.nodes)
        connections = [(k, v.weight, v.enabled) for k, v in genome.connections.items()]
        enabled_conns = [c for c in connections if c[2]]

        return {
            'num_nodes': n_nodes,
            'num_connections': len(enabled_conns),
            'total_connections': len(connections),
            'node_keys': list(genome.nodes.keys()),
            'connections': [(c[0], c[1]) for c in enabled_conns],
        }


# ======================================================================
if __name__ == "__main__":
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    loader.summary(X_train, X_test, y_train, y_test)

    print("\n--- NEAT EVOLUTION ---")
    engine = NeatEngine()
    best_genome, history = engine.run(X_train, y_train, generations=50)

    # Evaluate on test set
    results = engine.evaluate_best(X_test, y_test)
    topo = engine.get_topology_info()

    print(f"\n--- TEST RESULTS ---")
    print(results['report'])
    print("Confusion matrix:")
    print(results['confusion_matrix'])
    print(f"F1 Score (test): {results['f1']:.4f}")
    print(f"\n--- TOPOLOGY ---")
    print(f"  Nodes: {topo['num_nodes']}")
    print(f"  Connections: {topo['num_connections']} (enabled) / {topo['total_connections']} (total)")
