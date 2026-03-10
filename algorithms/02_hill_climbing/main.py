"""
Hill Climbing Algorithm — Perceptron Optimization for Exoplanet Classification.
Optimizes weights and bias by random perturbations.
Visualizes learning process and metrics in real-time using Pygame.
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

# --- Perceptron Model ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_proba(X, W, b):
    return sigmoid(np.dot(X, W) + b)

def calculate_fitness(y_true, y_pred_proba):
    # Accuracy is the fitness for Hill Climbing in this context
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return np.mean(y_true == y_pred)

# --- Pygame Visualizer Class ---
class HillClimbingVisualizer:
    def __init__(self, width=1000, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("IA Hill Climbing - Apprentissage")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.label_font = pygame.font.SysFont("Arial", 16)
        
        self.fitness_history = []
        self.max_history = 400
        
    def render(self, iteration, fitness, weights, bias, best_fitness):
        """Update display with current learning state."""
        self.screen.fill((15, 15, 25))
        
        # 1. Draw Title
        title = self.font.render(f"Hill Climbing - Itération {iteration}", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # 2. Draw Fitness Curve
        self.fitness_history.append(fitness)
        if len(self.fitness_history) > self.max_history:
            self.fitness_history.pop(0)
            
        curve_rect = pygame.Rect(50, 100, 400, 300)
        pygame.draw.rect(self.screen, (30, 30, 45), curve_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), curve_rect, 1)
        
        if len(self.fitness_history) > 1:
            points = []
            for i, val in enumerate(self.fitness_history):
                px = curve_rect.left + (i / self.max_history) * curve_rect.width
                # Scale fitness 0-1 to plot height
                py = curve_rect.bottom - (val * curve_rect.height)
                points.append((px, py))
            pygame.draw.lines(self.screen, (52, 152, 219), False, points, 2)
            
        fit_label = self.label_font.render(f"Fitness (Accuracy): {fitness:.4f}", True, (52, 152, 219))
        self.screen.blit(fit_label, (50, 410))
        best_label = self.label_font.render(f"Best: {best_fitness:.4f}", True, (46, 204, 113))
        self.screen.blit(best_label, (50, 435))

        # 3. Draw Weight Bars
        bar_area = pygame.Rect(500, 100, 450, 300)
        pygame.draw.rect(self.screen, (30, 30, 45), bar_area)
        pygame.draw.line(self.screen, (100, 100, 100), (bar_area.left, bar_area.centery), (bar_area.right, bar_area.centery), 1)
        
        n_weights = len(weights)
        bar_width = bar_area.width / (n_weights + 1)
        
        for i, w in enumerate(weights):
            hb = w * 50 # Scale weights for display
            bx = bar_area.left + (i+0.5) * bar_width
            by = bar_area.centery
            
            color = (52, 152, 219) if w > 0 else (231, 76, 60)
            rect = pygame.Rect(bx - bar_width/3, by - hb if w > 0 else by, bar_width/1.5, abs(hb))
            pygame.draw.rect(self.screen, color, rect)
            
            # Label
            w_label = self.label_font.render(f"W{i}", True, (150, 150, 150))
            self.screen.blit(w_label, (bx - 10, bar_area.bottom + 5))
            
        # Draw Bias
        bb_hb = bias * 50
        bb_bx = bar_area.left + n_weights * bar_width + bar_width/2
        bb_color = (241, 196, 15)
        bb_rect = pygame.Rect(bb_bx - bar_width/3, bar_area.centery - bb_hb if bias > 0 else bar_area.centery, bar_width/1.5, abs(bb_hb))
        pygame.draw.rect(self.screen, bb_color, bb_rect)
        bb_label = self.label_font.render("Bias", True, bb_color)
        self.screen.blit(bb_label, (bb_bx - 10, bar_area.bottom + 5))

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()

# --- Main Logic ---
def main(no_gui=False):
    # 1. Environment and Data
    np.random.seed(SEED)
    data = load_and_prepare_data()
    n_features = data.X_train.shape[1]
    
    # 2. Initialization
    W = np.random.randn(n_features) * 0.01
    b = 0.0
    
    current_proba_train = predict_proba(data.X_train, W, b)
    best_fitness = calculate_fitness(data.y_train, current_proba_train)
    
    best_W = W.copy()
    best_b = b
    
    # HC Parameters
    MAX_ITERS = 5000
    PERTURB_STRENGTH = 0.05
    STABILIZATION_WINDOW = 500
    improvement_history = []
    
    viz = None
    if not no_gui:
        viz = HillClimbingVisualizer()

    # 3. Hill Climbing Loop
    print("\n  [Hill Climbing] Début de l'optimisation...")
    
    with ResourceMonitor() as monitor:
        for i in range(MAX_ITERS):
            # Perturb
            perturb_W = np.random.randn(n_features) * PERTURB_STRENGTH
            perturb_b = np.random.randn() * PERTURB_STRENGTH
            
            temp_W = best_W + perturb_W
            temp_b = best_b + perturb_b
            
            # Eval
            temp_proba_train = predict_proba(data.X_train, temp_W, temp_b)
            temp_fitness = calculate_fitness(data.y_train, temp_proba_train)
            
            # Acceptance
            if temp_fitness > best_fitness:
                best_fitness = temp_fitness
                best_W = temp_W
                best_b = temp_b
                improvement_history.append(best_fitness)
            else:
                improvement_history.append(best_fitness)
            
            # GUI Update (limited frequency for speed)
            if not no_gui and i % 10 == 0:
                if not viz.render(i, best_fitness, best_W, best_b, best_fitness):
                    break
                    
            # Stabilization Check
            if len(improvement_history) > STABILIZATION_WINDOW:
                window = improvement_history[-STABILIZATION_WINDOW:]
                if window[-1] - window[0] < 1e-4: # Tolerance
                    print(f"  [Hill Climbing] Stabilisation atteinte à l'itération {i}")
                    break
        
        monitor.iterations = i

    if not no_gui:
        viz.close()

    # 4. Final Evaluation
    y_test_proba = predict_proba(data.X_test, best_W, best_b)
    metrics = compute_all_metrics(data.y_test, y_test_proba, DEFAULT_THRESHOLD)
    opt_threshold, j_stat = find_optimal_threshold(data.y_test, y_test_proba)
    
    # Use optimal threshold for unknown data
    y_unknown_proba = predict_proba(data.X_unknown, best_W, best_b)
    
    # 5. Save Results
    save_metrics(metrics, "02_hill_climbing", METRICS_DIR)
    save_resource_stats(monitor.get_stats(), "02_hill_climbing", METRICS_DIR)
    save_predictions(y_unknown_proba, "02_hill_climbing", PREDICTIONS_DIR, threshold=opt_threshold)
    
    # 6. Summary
    print_metrics_summary(metrics, "Hill Climbing")
    print(f"Seuil optimal suggéré : {opt_threshold:.4f} (J={j_stat:.4f})")
    print(f"Meilleure fitness train : {best_fitness:.4f}")

if __name__ == "__main__":
    headless = "--no-gui" in sys.argv
    main(no_gui=headless)
