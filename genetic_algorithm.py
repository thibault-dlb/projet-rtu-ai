"""
genetic_algorithm.py
Genetic Algorithm for binary classification of exoplanets.

Uses a population of simple perceptrons (weights + bias) evolved via:
  - Tournament selection
  - Uniform crossover
  - Gaussian mutation
  - Elitism
"""

import numpy as np
import time
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from data_loader import ExoplanetDataLoader


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class Individual:
    """A single perceptron individual in the population."""

    def __init__(self, n_features=12, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.weights = rng.standard_normal(n_features) * 0.5
        self.bias = rng.standard_normal() * 0.1
        self.fitness = 0.0

    def predict(self, X, threshold=0.5):
        z = X @ self.weights + self.bias
        return (sigmoid(z) >= threshold).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        self.fitness = f1_score(y, y_pred, zero_division=0)
        return self.fitness

    def copy(self):
        clone = Individual.__new__(Individual)
        clone.weights = self.weights.copy()
        clone.bias = self.bias
        clone.fitness = self.fitness
        return clone


class GeneticAlgorithm:
    """Population-based evolutionary optimizer for binary classification."""

    def __init__(self, pop_size=50, n_features=12,
                 mutation_rate=0.1, mutation_scale=0.2,
                 crossover_rate=0.7, elite_ratio=0.1,
                 seed=42):
        self.pop_size = pop_size
        self.n_features = n_features
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(pop_size * elite_ratio))
        self.rng = np.random.default_rng(seed)
        self.population = []
        self.best_individual = None

    # ------------------------------------------------------------------
    def initialize(self):
        """Create random initial population."""
        self.population = [
            Individual(self.n_features, self.rng) for _ in range(self.pop_size)
        ]

    # ------------------------------------------------------------------
    def evaluate_population(self, X, y):
        """Compute fitness for all individuals."""
        for ind in self.population:
            ind.evaluate(X, y)
        # Sort by fitness (descending)
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        self.best_individual = self.population[0].copy()

    # ------------------------------------------------------------------
    def _tournament_select(self, k=3):
        """Select one parent via tournament selection."""
        candidates = self.rng.choice(self.population, size=k, replace=False)
        return max(candidates, key=lambda i: i.fitness)

    # ------------------------------------------------------------------
    def _crossover(self, parent1, parent2):
        """Uniform crossover of weights and bias."""
        child = Individual.__new__(Individual)
        mask = self.rng.random(self.n_features) < 0.5
        child.weights = np.where(mask, parent1.weights, parent2.weights)
        child.bias = parent1.bias if self.rng.random() < 0.5 else parent2.bias
        child.fitness = 0.0
        return child

    # ------------------------------------------------------------------
    def _mutate(self, individual):
        """Gaussian noise mutation on weights and bias."""
        mutation_mask = self.rng.random(self.n_features) < self.mutation_rate
        noise = self.rng.standard_normal(self.n_features) * self.mutation_scale
        individual.weights += mutation_mask * noise

        if self.rng.random() < self.mutation_rate:
            individual.bias += self.rng.standard_normal() * self.mutation_scale

    # ------------------------------------------------------------------
    def evolve(self):
        """Perform one generation: selection, crossover, mutation, elitism."""
        # Elitism — keep top individuals
        new_pop = [ind.copy() for ind in self.population[:self.elite_count]]

        while len(new_pop) < self.pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if self.rng.random() < self.crossover_rate:
                child = self._crossover(p1, p2)
            else:
                child = p1.copy()

            self._mutate(child)
            new_pop.append(child)

        self.population = new_pop

    # ------------------------------------------------------------------
    def run(self, X_train, y_train, generations=100, callback=None):
        """Full training loop. Returns history dict."""
        self.initialize()

        history = {
            'best_fitness': [],
            'avg_fitness': [],
        }

        start_time = time.time()

        for gen in range(generations):
            self.evaluate_population(X_train, y_train)

            best_f = self.population[0].fitness
            avg_f = np.mean([ind.fitness for ind in self.population])

            history['best_fitness'].append(best_f)
            history['avg_fitness'].append(avg_f)

            if callback:
                callback(gen, {
                    'best_fitness': best_f,
                    'avg_fitness': avg_f,
                    'iteration': gen,
                    'elapsed_time': time.time() - start_time,
                })

            if (gen + 1) % 10 == 0:
                print(f"  Gen {gen+1:>4d} | Best: {best_f:.4f} | Avg: {avg_f:.4f}")

            self.evolve()

        # Final evaluation
        self.evaluate_population(X_train, y_train)
        history['best_individual'] = self.best_individual

        return history


# ======================================================================
if __name__ == "__main__":
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    loader.summary(X_train, X_test, y_train, y_test)

    print("\n--- GENETIC ALGORITHM ---")
    ga = GeneticAlgorithm(pop_size=50, mutation_rate=0.15, mutation_scale=0.3, seed=42)
    history = ga.run(X_train, y_train, generations=100)

    best = history['best_individual']
    y_pred = best.predict(X_test)

    print(f"\n--- TEST RESULTS ---")
    print(classification_report(y_test, y_pred, target_names=["Other", "Planet"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"F1 Score (test): {f1_score(y_test, y_pred):.4f}")
