"""
NEAT Algorithm — NeuroEvolution of Augmenting Topologies for Exoplanet Classification.
Evolves both weights and neural network structure.
Visualizes the best genome's topology in real-time using Pygame.
"""
import os
import sys
import time
import pickle
import numpy as np
import pygame
import neat

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

# --- Pygame Visualizer Class ---
class NEATVisualizer:
    def __init__(self, width=1000, height=700):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("IA NEAT - Évolution de Topologie")
        self.font = pygame.font.SysFont("Arial", 22)
        self.label_font = pygame.font.SysFont("Arial", 14)
        
    def render(self, generation, best_genome, config, info_text=""):
        """Draw the neural network topology of the best genome."""
        self.screen.fill((20, 15, 25)) # Dark purple tint
        
        # 1. Draw Title
        title = self.font.render(f"NEAT - Génération {generation}", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        if info_text:
            info = self.label_font.render(info_text, True, (200, 200, 200))
            self.screen.blit(info, (20, 50))
            
        # 2. Layout Nodes
        # neat-python uses negative IDs for inputs, positive for outputs/hidden
        # We need to map genome.nodes and genome.connections
        
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        
        # Find hidden nodes (nodes that are NOT inputs and NOT outputs)
        hidden_nodes = [node_id for node_id in best_genome.nodes.keys() if node_id not in output_nodes]
        hidden_nodes = [n for n in hidden_nodes if n not in input_nodes]
        
        # Positions mapping
        node_pos = {}
        
        # Inputs (Left column)
        for i, node_id in enumerate(input_nodes):
            node_pos[node_id] = (100, 100 + i * (self.height - 200) / (len(input_nodes) - 1 if len(input_nodes) > 1 else 1))
            
        # Outputs (Right column)
        for i, node_id in enumerate(output_nodes):
            node_pos[node_id] = (self.width - 100, self.height / 2)
            
        # Hidden (Middle - simple grid for now)
        if hidden_nodes:
            for i, node_id in enumerate(hidden_nodes):
                node_pos[node_id] = (self.width / 2, 100 + i * (self.height - 200) / (len(hidden_nodes) - 1 if len(hidden_nodes) > 1 else 1))

        # 3. Draw Connections
        for cg in best_genome.connections.values():
            if not cg.enabled: continue
            
            in_id, out_id = cg.key
            if in_id in node_pos and out_id in node_pos:
                start = node_pos[in_id]
                end = node_pos[out_id]
                
                # Color based on weight: Green positive, Red negative
                color = (46, 204, 113) if cg.weight > 0 else (231, 76, 60)
                width = int(abs(cg.weight) * 1.5) + 1
                width = min(10, width)
                
                pygame.draw.line(self.screen, color, start, end, width)

        # 4. Draw Nodes
        for node_id, pos in node_pos.items():
            color = (100, 100, 100) # Hidden (Gray)
            if node_id in input_nodes: color = (52, 152, 219) # Input (Blue)
            if node_id in output_nodes: color = (231, 76, 60) # Output (Red)
            
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 10)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(pos[0]), int(pos[1])), 10, 1)
            
            # Label
            label = self.label_font.render(str(node_id), True, (255, 255, 255))
            self.screen.blit(label, (pos[0] + 12, pos[1] - 8))

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()

# --- NEAT Logic ---
DATA = None # Global for fitness func

def eval_genomes(genomes, config):
    global DATA
    
    # We want to find the best for viz, but NEAT calls this per generation
    best_gen_fitness = -1
    best_gen_genome = None
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Evaluate on training data
        correct = 0
        total = len(DATA.y_train)
        
        # Batch evaluation (faster)
        # However, neat.nn.FeedForwardNetwork expects individual calls
        # To speed up, we can use a subset or just accept the latency
        # For this project, let's use the whole set but we note it's slow
        
        predictions = []
        for inputs in DATA.X_train:
            output = net.activate(inputs)
            predictions.append(output[0])
        
        predictions = np.array(predictions)
        accuracy = np.mean((predictions >= 0.5) == DATA.y_train)
        
        genome.fitness = accuracy
        
        if accuracy > best_gen_fitness:
            best_gen_fitness = accuracy
            best_gen_genome = genome

    # Global tracking for visualizer is handled by Repoters in neat-python
    # But since we want real-time GUI, we'll use a hack or a custom reporter
    pass

class PygameReporter(neat.reporting.BaseReporter):
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.generation = 0
        
    def start_generation(self, generation):
        self.generation = generation
        
    def post_evaluate(self, config, population, species, best_genome):
        info = f"Fitness Meilleur Individu: {best_genome.fitness:.4f} | Espèces: {len(species.species)}"
        if self.visualizer:
            self.visualizer.render(self.generation, best_genome, config, info)

def main(no_gui=False, generations=None, pop_size=None):
    global DATA
    DATA = load_and_prepare_data()
    
    # Load config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Overrides from config or args
    hp = ALGO_HYPERPARAMS["04_neat"]
    MAX_GENS = generations if generations is not None else hp["generations"]
    if pop_size is not None:
        config.pop_size = pop_size

    # Initialize Visualizer
    viz = None
    if not no_gui:
        viz = NEATVisualizer()

    # Create population
    p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(PygameReporter(viz))

    print(f"\n  [NEAT] Début de l'évolution ({MAX_GENS} gén, pop {config.pop_size})...")
    
    # Run
    # NEAT manages its own sessions, so we wrap the whole run in ResourceMonitor
    with ResourceMonitor() as monitor:
        winner = p.run(eval_genomes, MAX_GENS)
        monitor.iterations = p.generation

    if not no_gui:
        viz.close()

    # 5. Final Evaluation
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Predict on test
    test_probs = []
    for inputs in DATA.X_test:
        test_probs.append(winner_net.activate(inputs)[0])
    test_probs = np.array(test_probs)
    
    metrics = compute_all_metrics(DATA.y_test, test_probs, DEFAULT_THRESHOLD)
    opt_threshold, _ = find_optimal_threshold(DATA.y_test, test_probs)
    
    # Predict on unknown
    unknown_probs = []
    for inputs in DATA.X_unknown:
        unknown_probs.append(winner_net.activate(inputs)[0])
    unknown_probs = np.array(unknown_probs)
    
    # 6. Save Results
    save_metrics(metrics, "04_neat", METRICS_DIR)
    save_resource_stats(monitor.get_stats(), "04_neat", METRICS_DIR)
    save_predictions(unknown_probs, "04_neat", PREDICTIONS_DIR, threshold=opt_threshold)
    
    # Save winner model
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, '04_neat_winner.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    
    # 7. Summary
    print_metrics_summary(metrics, "NEAT")
    print(f"Meilleure fitness train : {winner.fitness:.4f}")
    print(f"Topologie : {len(winner.nodes)} nodes, {len(winner.connections)} connections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEAT Exoplanet Classifier")
    parser.add_argument("--no-gui", action="store_true", help="Run without Pygame visualization")
    parser.add_argument("--generations", type=int, help="Number of generations for evolution")
    parser.add_argument("--pop-size", type=int, help="Population size")
    args = parser.parse_args()
    
    main(no_gui=args.no_gui, generations=args.generations, pop_size=args.pop_size)
