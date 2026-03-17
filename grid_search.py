"""
grid_search.py
Multi-processed Grid Search for hyperparameter optimization.
Uses 12 workers (target CPU Ryzen 5500) to saturate computation.
"""

import multiprocessing
import numpy as np
import time
from typing import List, Tuple, Callable, Optional
from ai_engine import AIEngine, AlgorithmType
from data_loader import ExoplanetDataLoader

def run_single_point(args):
    """Worker function for multiprocessing."""
    algo_type, pop_size, gens, X_train, y_train, X_test, y_test = args
    engine = AIEngine(algo_type, pop_size=pop_size, generations=gens)
    try:
        # For Grid Search, we just need the final fitness
        res = engine.train(X_train, y_train, X_test, y_test)
        return (pop_size, gens, res.best_fitness)
    except Exception as e:
        print(f"Grid point ({pop_size}, {gens}) failed: {e}")
        return (pop_size, gens, 0.0)

class GridSearch:
    def __init__(self, algo_type: AlgorithmType, n_workers: int = 12):
        self.algo_type = algo_type
        self.n_workers = n_workers
        self.loader = ExoplanetDataLoader()
        self.results = None # Will be a 2D array (mesh)

    def run(self, pop_range: List[int], gen_range: List[int], callback: Optional[Callable] = None):
        """
        Run the grid search over pop_range and gen_range.
        pop_range: X values (e.g. [10, 20, 50, 100])
        gen_range: Y values (e.g. [5, 10, 20, 50])
        """
        X_train, X_test, y_train, y_test = self.loader.load_known()
        
        # Prepare argument list
        tasks = []
        for gens in gen_range:
            for pop in pop_range:
                tasks.append((self.algo_type, pop, gens, X_train, y_train, X_test, y_test))
        
        total_tasks = len(tasks)
        results_list = []
        
        start_time = time.time()
        
        # Use Pool for parallel execution
        with multiprocessing.Pool(processes=self.n_workers) as pool:
            # We use imap_unordered for potential speed and progress tracking
            for i, result in enumerate(pool.imap_unordered(run_single_point, tasks)):
                results_list.append(result)
                if callback:
                    callback(i + 1, total_tasks, result)
        
        # Convert to matrix
        z_values = np.zeros((len(gen_range), len(pop_range)))
        
        # Build mapping for quick lookup
        pop_map = {val: idx for idx, val in enumerate(pop_range)}
        gen_map = {val: idx for idx, val in enumerate(gen_range)}
        
        for pop, gens, fitness in results_list:
            z_values[gen_map[gens], pop_map[pop]] = fitness
            
        self.results = {
            'x': np.array(pop_range),
            'y': np.array(gen_range),
            'z': z_values,
            'time': time.time() - start_time
        }
        return self.results

if __name__ == "__main__":
    # Test Grid Search (small scale)
    gs = GridSearch(AlgorithmType.GA, n_workers=4)
    def test_cb(current, total, point):
        print(f"Progress: {current}/{total} | {point}")
        
    print("Running test grid search...")
    res = gs.run(pop_range=[10, 20, 30], gen_range=[5, 10, 15], callback=test_cb)
    print(f"Completed in {res['time']:.2f}s")
    print("Result Z matrix:\n", res['z'])
