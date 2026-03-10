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
import argparse

# Set path to root for shared module access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.config import (
    SEED, DEFAULT_THRESHOLD, RESULTS_DIR, 
    METRICS_DIR, PREDICTIONS_DIR, ALGO_COLORS,
    ALGO_HYPERPARAMS
)
from shared.data_loader import load_and_prepare_data
from shared.metrics import (
    compute_all_metrics, find_optimal_threshold, 
    save_metrics, save_predictions, print_metrics_summary
)
from shared.resource_monitor import ResourceMonitor, save_resource_stats

# --- Model Logic ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_proba(X, W, b):
    return sigmoid(np.dot(X, W) + b)

def calculate_fitness(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return np.mean(y_true == y_pred)

class HillClimbingRunner:
    def __init__(self, data, iterations=None, perturb_strength=None):
        self.data = data
        n_features = data.X_train.shape[1]
        
        # Hyperparams
        hp = ALGO_HYPERPARAMS["02_hill_climbing"]
        self.max_iters = iterations if iterations is not None else hp["iterations"]
        self.perturb_strength = perturb_strength if perturb_strength is not None else hp["perturb_strength"]
        
        # State
        self.W = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.best_W = self.W.copy()
        self.best_b = self.b
        
        current_proba_train = predict_proba(data.X_train, self.W, self.b)
        self.best_fitness = calculate_fitness(data.y_train, current_proba_train)
        
        self.iteration = 0
        self.fitness_history = []
        self.improvement_history = []
        self.running = True
        self.stabilization_window = 500
        self.monitor = ResourceMonitor()

    def start(self):
        self.monitor.__enter__()

    def step(self):
        if self.iteration >= self.max_iters or not self.running:
            self.running = False
            return False

        # Perturb
        perturb_W = np.random.randn(len(self.best_W)) * self.perturb_strength
        perturb_b = np.random.randn() * self.perturb_strength
        
        temp_W = self.best_W + perturb_W
        temp_b = self.best_b + perturb_b
        
        # Eval
        temp_proba_train = predict_proba(self.data.X_train, temp_W, temp_b)
        temp_fitness = calculate_fitness(self.data.y_train, temp_proba_train)
        
        # Acceptance
        if temp_fitness > self.best_fitness:
            self.best_fitness = temp_fitness
            self.best_W = temp_W
            self.best_b = temp_b

        self.fitness_history.append(self.best_fitness)
        self.improvement_history.append(self.best_fitness)
        
        # Stabilization Check
        if len(self.improvement_history) > self.stabilization_window:
            window = self.improvement_history[-self.stabilization_window:]
            if window[-1] - window[0] < 1e-4:
                self.running = False
                print(f"  [Hill Climbing] Stabilisation atteinte à l'itération {self.iteration}")

        self.iteration += 1
        return True

    def finish(self):
        self.monitor.__exit__(None, None, None)
        self.monitor.iterations = self.iteration
        
        # Evaluation
        y_test_proba = predict_proba(self.data.X_test, self.best_W, self.best_b)
        metrics = compute_all_metrics(self.data.y_test, y_test_proba, DEFAULT_THRESHOLD)
        opt_threshold, j_stat = find_optimal_threshold(self.data.y_test, y_test_proba)
        
        save_metrics(metrics, "02_hill_climbing", METRICS_DIR)
        save_resource_stats(self.monitor.get_stats(), "02_hill_climbing", METRICS_DIR)
        
        y_unknown_proba = predict_proba(self.data.X_unknown, self.best_W, self.best_b)
        save_predictions(y_unknown_proba, "02_hill_climbing", PREDICTIONS_DIR, threshold=opt_threshold)
        
        return metrics, opt_threshold

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill((15, 15, 25))
        
        # Font
        font = pygame.font.SysFont("Arial", 18)
        
        # Fitness Curve
        curve_rect = pygame.Rect(40, 60, width//2 - 60, height - 120)
        pygame.draw.rect(surface, (30, 30, 45), curve_rect)
        pygame.draw.rect(surface, (100, 100, 100), curve_rect, 1)
        
        hist = self.fitness_history[-400:]
        if len(hist) > 1:
            points = []
            for i, val in enumerate(hist):
                px = curve_rect.left + (i / max(1, len(hist)-1)) * curve_rect.width
                py = curve_rect.bottom - (val * curve_rect.height)
                points.append((px, py))
            pygame.draw.lines(surface, (52, 152, 219), False, points, 2)
            
        fit_txt = font.render(f"Best Fitness: {self.best_fitness:.4f}", True, (255, 255, 255))
        surface.blit(fit_txt, (curve_rect.x, curve_rect.y - 30))

        # Weight Bars
        bar_area = pygame.Rect(width//2 + 20, 60, width//2 - 60, height - 120)
        pygame.draw.rect(surface, (30, 30, 45), bar_area)
        pygame.draw.line(surface, (100, 100, 100), (bar_area.left, bar_area.centery), (bar_area.right, bar_area.centery), 1)
        
        n_w = len(self.best_W)
        bw = bar_area.width / (n_w + 1)
        for i, w in enumerate(self.best_W):
            h = w * 100
            bx = bar_area.left + (i+0.5) * bw
            color = (52, 152, 219) if w > 0 else (231, 76, 60)
            r = pygame.Rect(bx - bw/3, bar_area.centery - h if w > 0 else bar_area.centery, bw/1.5, abs(h))
            pygame.draw.rect(surface, color, r)

# --- compatibility main ---
def main(no_gui=False, iterations=None):
    np.random.seed(SEED)
    data = load_and_prepare_data()
    runner = HillClimbingRunner(data, iterations=iterations)
    runner.start()
    
    if no_gui:
        while runner.step(): pass
    else:
        pygame.init()
        screen = pygame.display.set_mode((1000, 600))
        while runner.running:
            runner.step()
            runner.draw(screen)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: runner.running = False
        pygame.quit()
        
    runner.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    main(no_gui=args.no_gui, iterations=args.iterations)
