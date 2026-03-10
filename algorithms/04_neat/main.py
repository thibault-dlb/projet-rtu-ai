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

def eval_genomes(genomes, config, data):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        predictions = []
        for inputs in data.X_train:
            output = net.activate(inputs)
            predictions.append(output[0])
        predictions = np.array(predictions)
        accuracy = np.mean((predictions >= 0.5) == data.y_train)
        genome.fitness = accuracy

class NEATRunner:
    def __init__(self, data, generations=None, pop_size=None):
        self.data = data
        
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-neat.txt')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

        hp = ALGO_HYPERPARAMS["04_neat"]
        self.max_gens = generations if generations is not None else hp["generations"]
        if pop_size is not None:
            self.config.pop_size = pop_size

        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        self.generation = 0
        self.best_genome = None
        self.running = True
        self.monitor = ResourceMonitor()

    def start(self):
        self.monitor.__enter__()

    def step(self):
        if self.generation >= self.max_gens or not self.running:
            self.running = False
            return False

        # Run 1 generation
        self.best_genome = self.population.run(lambda genomes, config: eval_genomes(genomes, config, self.data), 1)
        self.generation = self.population.generation
        return True

    def finish(self):
        self.monitor.__exit__(None, None, None)
        self.monitor.iterations = self.generation
        
        winner_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        
        test_probs = []
        for inputs in self.data.X_test:
            test_probs.append(winner_net.activate(inputs)[0])
        test_probs = np.array(test_probs)
        
        metrics = compute_all_metrics(self.data.y_test, test_probs, DEFAULT_THRESHOLD)
        opt_threshold, _ = find_optimal_threshold(self.data.y_test, test_probs)
        
        save_metrics(metrics, "04_neat", METRICS_DIR)
        save_resource_stats(self.monitor.get_stats(), "04_neat", METRICS_DIR)
        
        unknown_probs = []
        for inputs in self.data.X_unknown:
            unknown_probs.append(winner_net.activate(inputs)[0])
        unknown_probs = np.array(unknown_probs)
        save_predictions(unknown_probs, "04_neat", PREDICTIONS_DIR, threshold=opt_threshold)
        
        # Save winner
        with open(os.path.join(METRICS_DIR, '04_neat_winner.pkl'), 'wb') as f:
            pickle.dump(self.best_genome, f)
            
        return metrics, opt_threshold

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill((20, 15, 25))
        
        if not self.best_genome: return
        
        font = pygame.font.SysFont("Arial", 16)
        
        input_nodes = self.config.genome_config.input_keys
        output_nodes = self.config.genome_config.output_keys
        hidden_nodes = [node_id for node_id in self.best_genome.nodes.keys() if node_id not in output_nodes and node_id not in input_nodes]
        
        pos = {}
        for i, nid in enumerate(input_nodes):
            pos[nid] = (100, 100 + i * (height - 200) / (len(input_nodes) - 1 if len(input_nodes) > 1 else 1))
        for i, nid in enumerate(output_nodes):
            pos[nid] = (width - 100, height / 2)
        for i, nid in enumerate(hidden_nodes):
            pos[nid] = (width / 2, 100 + i * (height - 200) / (len(hidden_nodes) - 1 if len(hidden_nodes) > 1 else 1))

        # Connections
        for cg in self.best_genome.connections.values():
            if not cg.enabled: continue
            in_id, out_id = cg.key
            if in_id in pos and out_id in pos:
                color = (46, 204, 113) if cg.weight > 0 else (231, 76, 60)
                width_line = int(abs(cg.weight)) + 1
                pygame.draw.line(surface, color, pos[in_id], pos[out_id], min(5, width_line))

        # Nodes
        for nid, p in pos.items():
            color = (52, 152, 219) if nid in input_nodes else (231, 76, 60) if nid in output_nodes else (100, 100, 100)
            pygame.draw.circle(surface, color, (int(p[0]), int(p[1])), 8)
            pygame.draw.circle(surface, (255, 255, 255), (int(p[0]), int(p[1])), 8, 1)

# --- compatibility main ---
def main(no_gui=False, generations=None, pop_size=None):
    data = load_and_prepare_data()
    runner = NEATRunner(data, generations=generations, pop_size=pop_size)
    runner.start()
    
    if no_gui:
        while runner.step(): pass
    else:
        pygame.init()
        screen = pygame.display.set_mode((1000, 700))
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
