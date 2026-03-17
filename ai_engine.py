"""
ai_engine.py
Unified interface for all exoplanet classification algorithms.

Supported algorithms:
  - RANDOM: Unified random baseline.
  - HILL_CLIMBING: Single-individual evolutionary search.
  - GA: Population-based genetic algorithm.
  - NEAT: Topology-evolving neural networks.
"""

import numpy as np
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from data_loader import ExoplanetDataLoader
import baseline_random
import baseline_hill_climbing
from genetic_algorithm import GeneticAlgorithm, Individual
from neat_engine import NeatEngine
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score


class AlgorithmType(Enum):
    RANDOM = auto()
    HILL_CLIMBING = auto()
    GA = auto()
    NEAT = auto()


@dataclass
class TrainingResult:
    algorithm: AlgorithmType
    best_fitness: float = 0.0
    test_f1: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    history: Dict[str, List[float]] = field(default_factory=dict)
    topology_info: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    best_model: Any = None


class AIEngine:
    """Unified engine to train and evaluate any supported algorithm."""

    def __init__(self, algorithm_type: AlgorithmType, **params):
        self.algorithm_type = algorithm_type
        self.params = params
        self.loader = ExoplanetDataLoader()
        self.neat_engine = NeatEngine() if algorithm_type == AlgorithmType.NEAT else None
        self.best_model = None
        self.last_test_results = {}

    def train(self, X_train, y_train, X_test, y_test, callback: Optional[Callable] = None) -> TrainingResult:
        """Train the selected algorithm with an optional progress callback."""
        start_time = time.time()
        result = TrainingResult(algorithm=self.algorithm_type)

        if self.algorithm_type == AlgorithmType.RANDOM:
            # Random doesn't really train, just a placeholder run
            rng = np.random.default_rng(seed=self.params.get('seed', 42))
            y_pred = baseline_random.random_classify(X_test, rng)
            result.test_f1 = f1_score(y_test, y_pred)
            result.confusion_matrix = confusion_matrix(y_test, y_pred)

        elif self.algorithm_type == AlgorithmType.HILL_CLIMBING:
            iters = self.params.get('iterations', 1000)
            weights, bias, history = baseline_hill_climbing.hill_climbing(
                X_train, y_train, iterations=iters, seed=self.params.get('seed', 42)
            )
            self.best_model = {'weights': weights, 'bias': bias}
            result.best_fitness = history[-1]
            result.history = {'best_fitness': history}
            result.topology_info = {'weights_shape': weights.shape}

        elif self.algorithm_type == AlgorithmType.GA:
            ga = GeneticAlgorithm(
                pop_size=self.params.get('pop_size', 50),
                seed=self.params.get('seed', 42)
            )
            history = ga.run(
                X_train, y_train, 
                generations=self.params.get('generations', 100),
                callback=callback
            )
            self.best_model = history['best_individual']
            result.best_fitness = history['best_fitness'][-1]
            result.history = history
            result.topology_info = {'weights_shape': (12,)}

        elif self.algorithm_type == AlgorithmType.NEAT:
            # Respect user's laptop constraints if specified in params
            generations = self.params.get('generations', 50)
            if self.params.get('light_mode', False):
                self.neat_engine.config.pop_size = 20
                generations = min(generations, 10)
            
            best_genome, history = self.neat_engine.run(
                X_train, y_train, 
                generations=generations,
                callback=callback
            )
            self.best_model = best_genome
            result.best_fitness = history['best_fitness'][-1]
            result.history = history
            result.topology_info = self.neat_engine.get_topology_info(best_genome)

        result.training_time = time.time() - start_time
        
        # Evaluate on test set if not already done (RANDOM)
        if self.algorithm_type != AlgorithmType.RANDOM:
            eval_metrics = self.evaluate(X_test, y_test)
            result.test_f1 = eval_metrics['f1']
            result.confusion_matrix = eval_metrics['confusion_matrix']
            
        result.best_model = self.best_model
        return result

    def predict(self, X, threshold=0.5):
        """Predict outcomes for given input using the trained model."""
        if self.best_model is None and self.algorithm_type != AlgorithmType.RANDOM:
            raise ValueError("Model not trained yet.")

        if self.algorithm_type == AlgorithmType.RANDOM:
            rng = np.random.default_rng()
            return rng.integers(0, 2, size=len(X))

        if self.algorithm_type == AlgorithmType.HILL_CLIMBING:
            return baseline_hill_climbing.predict(
                X, self.best_model['weights'], self.best_model['bias'], threshold
            )

        if self.algorithm_type == AlgorithmType.GA:
            return self.best_model.predict(X, threshold)

        if self.algorithm_type == AlgorithmType.NEAT:
            return self.neat_engine.predict(self.best_model, X, threshold)

    def evaluate(self, X, y, threshold=0.5):
        """Compute comprehensive metrics for the current model."""
        y_pred = self.predict(X, threshold)
        
        metrics = {
            'f1': f1_score(y, y_pred, zero_division=0),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred
        }
        self.last_test_results = metrics
        return metrics

    def save_history(self, filepath="evolution_history.json"):
        """Save training history if supported by the algorithm."""
        if self.algorithm_type == AlgorithmType.NEAT and self.neat_engine:
            return self.neat_engine.save_history(filepath)
        return False


if __name__ == "__main__":
    # Test unified engine
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    
    # Run a quick test for each
    for algo in AlgorithmType:
        print(f"\n>>> TESTING ALGORITHM: {algo.name}")
        engine = AIEngine(algo, generations=5, iterations=50, pop_size=10, light_mode=True)
        
        def simple_callback(gen, metrics):
            if gen % 2 == 0:
                print(f"  Gen {gen}: Fitness={metrics['best_fitness']:.4f}")
                
        res = engine.train(X_train, y_train, X_test, y_test, callback=simple_callback)
        print(f"  Result: Test F1={res.test_f1:.4f}, Time={res.training_time:.2f}s")
