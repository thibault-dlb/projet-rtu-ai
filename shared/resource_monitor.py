"""
Resource monitoring module for the Exoplanet Classification project.
Measures execution time and peak memory usage.
"""
import os
import time
import json
import tracemalloc

class ResourceMonitor:
    """
    Context manager to track execution time and peak memory usage.
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.peak_memory = 0
        self.iterations = None
        
    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak
        tracemalloc.stop()
        
    def get_stats(self):
        """Returns recorded statistics as a dictionary."""
        return {
            "elapsed_time_sec": self.end_time - self.start_time,
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "iterations": self.iterations
        }

def save_resource_stats(stats, algo_name, results_dir):
    """Save resource usage statistics to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{algo_name}_resources.json")
    
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"  [Resources] Sauvegardées dans {file_path}")
