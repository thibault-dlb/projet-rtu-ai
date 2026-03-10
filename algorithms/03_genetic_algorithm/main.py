"""
Standard Genetic Algorithm — Weight Optimization for Exoplanet Classification.
Evolutionary approach with selection, crossover, and mutation.
Visualizes population dynamics and fitness progression using Pygame.
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

# --- Perceptron Model (Simple logic for GA) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_proba(X, weights_vector):
    W = weights_vector[:-1]
    b = weights_vector[-1]
    return sigmoid(np.dot(X, W) + b)

def calculate_fitness(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return np.mean(y_true == y_pred)

def tournament_selection(population, fitnesses, k=3):
    idx = np.random.randint(0, len(population), k)
    best_idx = idx[np.argmax(fitnesses[idx])]
    return population[best_idx].copy()

def crossover(parent1, parent2, rate=0.8):
    if np.random.rand() < rate:
        mask = np.random.rand(len(parent1)) > 0.5
        child = np.where(mask, parent1, parent2)
        return child
    return parent1.copy()

def mutate(individual, rate=0.1, sigma=0.2):
    mask = np.random.rand(len(individual)) < rate
    individual[mask] += np.random.randn(np.sum(mask)) * sigma
    return individual

class GARunner:
    def __init__(self, data, generations=None, pop_size=None):
        self.data = data
        n_params = data.X_train.shape[1] + 1
        
        hp = ALGO_HYPERPARAMS["03_genetic_algorithm"]
        self.pop_size = pop_size if pop_size is not None else hp["pop_size"]
        self.max_gens = generations if generations is not None else hp["generations"]
        self.mutation_rate = hp["mutation_rate"]
        self.mutation_sigma = 0.2
        self.elitism_count = 2
        
        self.population = np.random.randn(self.pop_size, n_params) * 0.1
        self.gen = 0
        self.best_history = []
        self.avg_history = []
        
        self.best_overall_ind = None
        self.best_overall_fit = -1
        self.running = True
        self.monitor = ResourceMonitor()

    def start(self):
        self.monitor.__enter__()

    def step(self):
        if self.gen >= self.max_gens or not self.running:
            self.running = False
            return False

        # Evaluate
        fitnesses = []
        for ind in self.population:
            y_proba = predict_proba(self.data.X_train, ind)
            fitnesses.append(calculate_fitness(self.data.y_train, y_proba))
        fitnesses = np.array(fitnesses)
        
        # Track best
        best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[best_idx]
        gen_avg_fit = np.mean(fitnesses)
        
        if gen_best_fit > self.best_overall_fit:
            self.best_overall_fit = gen_best_fit
            self.best_overall_ind = self.population[best_idx].copy()
            
        self.best_history.append(gen_best_fit)
        self.avg_history.append(gen_avg_fit)
        
        # New Population
        new_population = []
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(self.elitism_count):
            new_population.append(self.population[sorted_indices[i]])
            
        while len(new_population) < self.pop_size:
            p1 = tournament_selection(self.population, fitnesses)
            p2 = tournament_selection(self.population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, rate=self.mutation_rate, sigma=self.mutation_sigma)
            new_population.append(child)
            
        self.population = np.array(new_population)
        self.gen += 1
        return True

    def finish(self):
        self.monitor.__exit__(None, None, None)
        self.monitor.iterations = self.gen
        
        y_test_proba = predict_proba(self.data.X_test, self.best_overall_ind)
        metrics = compute_all_metrics(self.data.y_test, y_test_proba, DEFAULT_THRESHOLD)
        opt_threshold, _ = find_optimal_threshold(self.data.y_test, y_test_proba)
        
        save_metrics(metrics, "03_genetic_algorithm", METRICS_DIR)
        save_resource_stats(self.monitor.get_stats(), "03_genetic_algorithm", METRICS_DIR)
        
        y_unknown_proba = predict_proba(self.data.X_unknown, self.best_overall_ind)
        save_predictions(y_unknown_proba, "03_genetic_algorithm", PREDICTIONS_DIR, threshold=opt_threshold)
        
        return metrics, opt_threshold

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill((15, 20, 15))
        
        font = pygame.font.SysFont("Arial", 18)
        
        # History Chart
        chart_rect = pygame.Rect(40, 60, width//2 -60, height - 120)
        pygame.draw.rect(surface, (20, 35, 20), chart_rect)
        pygame.draw.rect(surface, (100, 120, 100), chart_rect, 1)
        
        hist_b = self.best_history[-200:]
        hist_a = self.avg_history[-200:]
        if len(hist_b) > 1:
            best_pts = []
            avg_pts = []
            for i, (b, a) in enumerate(zip(hist_b, hist_a)):
                px = chart_rect.left + (i / max(1, len(hist_b)-1)) * chart_rect.width
                py_b = chart_rect.bottom - (b * chart_rect.height)
                py_a = chart_rect.bottom - (a * chart_rect.height)
                best_pts.append((px, py_b))
                avg_pts.append((px, py_a))
            pygame.draw.lines(surface, (46, 204, 113), False, best_pts, 3)
            pygame.draw.lines(surface, (52, 152, 219), False, avg_pts, 2)
            
        l_best = font.render(f"Best Fitness: {self.best_overall_fit:.4f}", True, (46, 204, 113))
        surface.blit(l_best, (chart_rect.x, chart_rect.y - 30))

        # Scatter (Diversity)
        scatter_rect = pygame.Rect(width//2 + 20, 60, width//2 - 60, height - 120)
        pygame.draw.rect(surface, (20, 35, 20), scatter_rect)
        pygame.draw.rect(surface, (100, 120, 100), scatter_rect, 1)
        pygame.draw.line(surface, (100, 100, 100), (scatter_rect.centerx, scatter_rect.top), (scatter_rect.centerx, scatter_rect.bottom), 1)
        pygame.draw.line(surface, (100, 100, 100), (scatter_rect.left, scatter_rect.centery), (scatter_rect.right, scatter_rect.centery), 1)
        
        for ind in self.population:
            sc_x = scatter_rect.centerx + ind[0] * 30
            sc_y = scatter_rect.centery - ind[1] * 30
            sc_x = max(scatter_rect.left + 5, min(scatter_rect.right - 5, sc_x))
            sc_y = max(scatter_rect.top + 5, min(scatter_rect.bottom - 5, sc_y))
            pygame.draw.circle(surface, (150, 200, 150), (int(sc_x), int(sc_y)), 2)

# --- compatibility main ---
def main(no_gui=False, generations=None, pop_size=None):
    np.random.seed(SEED)
    data = load_and_prepare_data()
    runner = GARunner(data, generations=generations, pop_size=pop_size)
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
    parser.add_argument("--generations", type=int)
    parser.add_argument("--pop-size", type=int)
    args = parser.parse_args()
    main(no_gui=args.no_gui, generations=args.generations, pop_size=args.pop_size)
