"""
Random Algorithm — Baseline for Exoplanet Classification.
Generates pure random predictions and visualizes them using Pygame.
"""
import os
import sys
import time
import numpy as np
import pygame

# Set path to root for shared module access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.config import (
    SEED, DEFAULT_THRESHOLD, RESULTS_DIR, 
    METRICS_DIR, PREDICTIONS_DIR, ALGO_COLORS
)
from shared.data_loader import load_and_prepare_data
from shared.metrics import (
    compute_all_metrics, find_optimal_threshold, 
    save_metrics, save_predictions, print_metrics_summary
)
from shared.resource_monitor import ResourceMonitor, save_resource_stats

# --- Pygame Visualizer Class ---
class RandomVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("IA Aléatoire - Visualisation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.points = [] # List of tuples (x, y, color)
        self.max_points = 1000
        
    def update(self, proba, actual=None):
        """Update visualization with a new prediction."""
        # Clean screen
        self.screen.fill((20, 20, 30))
        
        # Add new point
        x = int(proba * (self.width - 100)) + 50
        y = np.random.randint(150, self.height - 50)
        
        # Color based on truth if available
        color = (200, 200, 200) # Gray
        if actual is not None:
            color = (46, 204, 113) if actual == 1 else (231, 76, 60) # Green/Red
            
        self.points.append((x, y, color))
        if len(self.points) > self.max_points:
            self.points.pop(0)

        # Draw Title
        title = self.font.render("Algorithme Aléatoire (Baseline)", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Draw axes/threshold
        pygame.draw.line(self.screen, (100, 100, 100), (50, 100), (self.width-50, 100), 2)
        pygame.draw.line(self.screen, (255, 255, 0), (int(0.5*(self.width-100))+50, 80), (int(0.5*(self.width-100))+50, 120), 2)
        th_label = self.small_font.render("Seuil (0.5)", True, (255, 255, 0))
        self.screen.blit(th_label, (int(0.5*(self.width-100))+30, 60))

        # Draw points
        for p in self.points:
            pygame.draw.circle(self.screen, p[2], (p[0], p[1]), 3)

        # Draw summary
        summary = self.small_font.render(f"Dernière proba: {proba:.4f}", True, (255, 255, 255))
        self.screen.blit(summary, (20, self.height - 30))
        
        pygame.display.flip()
        
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def show_final_stats(self, metrics):
        """Show final metrics on screen."""
        self.screen.fill((20, 20, 30))
        lines = [
            "ENTRAÎNEMENT TERMINÉ",
            f"Accuracy : {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall   : {metrics['recall']:.4f}",
            f"F1-Score : {metrics['f1']:.4f}",
            "",
            "Appuyez sur une touche pour quitter..."
        ]
        curr_y = 150
        for line in lines:
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (100, curr_y))
            curr_y += 40
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
        pygame.quit()

# --- Main Execution ---
def main(no_gui=False):
    # 1. Load Data
    data = load_and_prepare_data()
    
    # Init Visualizer
    viz = None
    if not no_gui:
        try:
            viz = RandomVisualizer()
        except Exception as e:
            print(f"  [Warning] Pygame failed to initialize: {e}. Running in headless mode.")
            no_gui = True

    # 2. Run "Training" (Resource Tracking)
    with ResourceMonitor() as monitor:
        # For random, training is just setting a seed
        np.random.seed(SEED)
        time.sleep(1) # Fake some work for visualization
        
        # Predict on Test Set (for metrics)
        y_test_proba = np.random.uniform(0.0, 1.0, size=len(data.y_test))
        
        # Predict on Unknown Set
        y_unknown_proba = np.random.uniform(0.0, 1.0, size=len(data.X_unknown))
        
        monitor.iterations = 1 # Not iterative

    # 3. Visualization loop (simulation)
    if not no_gui and viz:
        print("  [Visualisation] Lancement de l'animation...")
        # Simulate processing the test set
        for i in range(min(500, len(y_test_proba))):
            if not viz.update(y_test_proba[i], data.y_test[i]):
                break
            time.sleep(0.01)

    # 4. Evaluate
    metrics = compute_all_metrics(data.y_test, y_test_proba, DEFAULT_THRESHOLD)
    opt_threshold, j_stat = find_optimal_threshold(data.y_test, y_test_proba)
    
    # 5. Save Results
    res_stats = monitor.get_stats()
    save_metrics(metrics, "01_random", METRICS_DIR)
    save_resource_stats(res_stats, "01_random", METRICS_DIR)
    save_predictions(y_unknown_proba, "01_random", PREDICTIONS_DIR, threshold=opt_threshold)
    
    # 6. Final Summary
    print_metrics_summary(metrics, "Aléatoire")
    print(f"Seuil optimal suggéré : {opt_threshold:.4f} (J={j_stat:.4f})")
    
    if not no_gui and viz:
        viz.show_final_stats(metrics)

if __name__ == "__main__":
    headless = "--no-gui" in sys.argv
    main(no_gui=headless)
