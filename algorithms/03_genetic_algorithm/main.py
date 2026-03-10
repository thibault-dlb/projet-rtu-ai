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

# --- Perceptron Model (Simple logic for GA) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_proba(X, weights_vector):
    # weights_vector is [w1..w12, bias]
    W = weights_vector[:-1]
    b = weights_vector[-1]
    return sigmoid(np.dot(X, W) + b)

def calculate_fitness(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return np.mean(y_true == y_pred)

# --- Pygame Visualizer Class ---
class GAVisualizer:
    def __init__(self, width=1000, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("IA Algorithme Génétique - Évolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.label_font = pygame.font.SysFont("Arial", 16)
        
        self.best_history = []
        self.avg_history = []
        self.max_history = 100
        
    def render(self, generation, best_fit, avg_fit, population):
        """Update display with evolutionary state."""
        self.screen.fill((15, 20, 15)) # Dark green tint
        
        # 1. Draw Title
        title = self.font.render(f"Algorithme Génétique - Génération {generation}", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # 2. Draw History Chart
        self.best_history.append(best_fit)
        self.avg_history.append(avg_fit)
        if len(self.best_history) > self.max_history:
            self.best_history.pop(0)
            self.avg_history.pop(0)
            
        chart_rect = pygame.Rect(50, 100, 450, 300)
        pygame.draw.rect(self.screen, (20, 35, 20), chart_rect)
        pygame.draw.rect(self.screen, (100, 120, 100), chart_rect, 1)
        
        if len(self.best_history) > 1:
            best_pts = []
            avg_pts = []
            for i, (b, a) in enumerate(zip(self.best_history, self.avg_history)):
                px = chart_rect.left + (i / (self.max_history-1)) * chart_rect.width
                py_b = chart_rect.bottom - (b * chart_rect.height)
                py_a = chart_rect.bottom - (a * chart_rect.height)
                best_pts.append((px, py_b))
                avg_pts.append((px, py_a))
            pygame.draw.lines(self.screen, (46, 204, 113), False, best_pts, 3)
            pygame.draw.lines(self.screen, (52, 152, 219), False, avg_pts, 2)
            
        legend_b = self.label_font.render(f"Meilleur Fitness: {best_fit:.4f}", True, (46, 204, 113))
        legend_a = self.label_font.render(f"Moyenne Pop   : {avg_fit:.4f}", True, (52, 152, 219))
        self.screen.blit(legend_b, (50, 410))
        self.screen.blit(legend_a, (50, 435))

        # 3. Draw Population Scatter (Visualization of weight diversity)
        # Using weights 0 and 1 as coordinates
        scatter_rect = pygame.Rect(550, 100, 400, 300)
        pygame.draw.rect(self.screen, (20, 35, 20), scatter_rect)
        pygame.draw.rect(self.screen, (100, 120, 100), scatter_rect, 1)
        pygame.draw.line(self.screen, (100, 100, 100), (scatter_rect.centerx, scatter_rect.top), (scatter_rect.centerx, scatter_rect.bottom), 1)
        pygame.draw.line(self.screen, (100, 100, 100), (scatter_rect.left, scatter_rect.centery), (scatter_rect.right, scatter_rect.centery), 1)
        
        for ind in population:
            # Scale weights -5 to 5 to scatter area
            sc_x = scatter_rect.centerx + ind[0] * 40
            sc_y = scatter_rect.centery - ind[1] * 40 
            
            # Clip to rect
            sc_x = max(scatter_rect.left + 5, min(scatter_rect.right - 5, sc_x))
            sc_y = max(scatter_rect.top + 5, min(scatter_rect.bottom - 5, sc_y))
            
            pygame.draw.circle(self.screen, (150, 200, 150), (int(sc_x), int(sc_y)), 2)
            
        sc_label = self.label_font.render("Dispersion Population (W0 vs W1)", True, (200, 200, 200))
        self.screen.blit(sc_label, (550, 410))

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()

# --- GA Functions ---
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

# --- Main Logic ---
def main(no_gui=False):
    np.random.seed(SEED)
    data = load_and_prepare_data()
    n_params = data.X_train.shape[1] + 1 # 12 weights + 1 bias
    
    # GA Parameters
    POP_SIZE = 60
    MAX_GENS = 100
    MUTATION_RATE = 0.1
    MUTATION_SIGMA = 0.2
    ELITISM_COUNT = 2
    
    # Initialize Population
    population = np.random.randn(POP_SIZE, n_params) * 0.1
    
    viz = None
    if not no_gui:
        viz = GAVisualizer()

    best_overall_ind = None
    best_overall_fit = -1
    
    print("\n  [Algorithme Génétique] Début de l'évolution...")
    
    with ResourceMonitor() as monitor:
        for gen in range(MAX_GENS):
            # 1. Evaluate Fitness
            fitnesses = []
            for ind in population:
                y_proba = predict_proba(data.X_train, ind)
                fitnesses.append(calculate_fitness(data.y_train, y_proba))
            fitnesses = np.array(fitnesses)
            
            # 2. Track best
            best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[best_idx]
            gen_avg_fit = np.mean(fitnesses)
            
            if gen_best_fit > best_overall_fit:
                best_overall_fit = gen_best_fit
                best_overall_ind = population[best_idx].copy()
            
            # 3. GUI Update
            if not no_gui:
                if not viz.render(gen, gen_best_fit, gen_avg_fit, population):
                    break
                time.sleep(0.01) # Small delay to see progress

            # 4. New Population
            new_population = []
            
            # Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(ELITISM_COUNT):
                new_population.append(population[sorted_indices[i]])
                
            # Fill remaining
            while len(new_population) < POP_SIZE:
                p1 = tournament_selection(population, fitnesses)
                p2 = tournament_selection(population, fitnesses)
                child = crossover(p1, p2)
                child = mutate(child, rate=MUTATION_RATE, sigma=MUTATION_SIGMA)
                new_population.append(child)
                
            population = np.array(new_population)
            
            # Stabilization Check (Optional)
            if gen > 20 and (best_overall_fit > 0.85): # Just an example break
                # Could check if avg and best are very close
                pass

        monitor.iterations = MAX_GENS

    if not no_gui:
        viz.close()

    # 5. Final Evaluation
    y_test_proba = predict_proba(data.X_test, best_overall_ind)
    metrics = compute_all_metrics(data.y_test, y_test_proba, DEFAULT_THRESHOLD)
    opt_threshold, _ = find_optimal_threshold(data.y_test, y_test_proba)
    
    y_unknown_proba = predict_proba(data.X_unknown, best_overall_ind)
    
    # 6. Save Results
    save_metrics(metrics, "03_genetic_algorithm", METRICS_DIR)
    save_resource_stats(monitor.get_stats(), "03_genetic_algorithm", METRICS_DIR)
    save_predictions(y_unknown_proba, "03_genetic_algorithm", PREDICTIONS_DIR, threshold=opt_threshold)
    
    # 7. Summary
    print_metrics_summary(metrics, "Algorithme Génétique")
    print(f"Meilleure fitness train : {best_overall_fit:.4f}")

if __name__ == "__main__":
    headless = "--no-gui" in sys.argv
    main(no_gui=headless)
