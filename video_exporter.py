"""
video_exporter.py
Manim-based video generator for neural network evolution.
Generates an animation showing nodes and connections evolving over time.
"""

from manim import *
import numpy as np
import json
import os

# Opera GX style colors
COLOR_BG = "#0f0f0f"
COLOR_ACCENT = "#ff3c3c" # Red
COLOR_CYAN = "#00ffff"
COLOR_HIDDEN = "#999999"

class EvolutionScene(Scene):
    def construct(self):
        self.camera.background_color = COLOR_BG
        
        # Load mock or real data
        # For the script to be runnable standalone, we'll look for a history file
        history_path = "evolution_history.json"
        if not os.path.exists(history_path):
            self.show_demo_text()
            return

        with open(history_path, 'r') as f:
            history = json.load(f)

        title = Text("Evolutionary Intelligence", color=COLOR_ACCENT).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Container for the current network mobjects
        current_net = VGroup()
        
        last_topology = None
        
        # We only animate a subset of generations to keep it fast
        # (e.g., first, middle, last)
        gen_indices = [0, len(history)//2, len(history)-1]
        # Remove duplicates
        gen_indices = sorted(list(set(gen_indices)))

        for i in gen_indices:
            frame = history[i]
            gen_label = Text(f"Generation {frame['iteration']}", font_size=24).next_to(title, DOWN)
            fitness_label = Text(f"Fitness: {frame['best_fitness']:.4f}", font_size=24, color=COLOR_CYAN).next_to(gen_label, DOWN)
            
            new_net = self.create_network_mobject(frame['topology'])
            
            if last_topology is None:
                self.play(
                    FadeIn(new_net),
                    Write(gen_label),
                    Write(fitness_label)
                )
            else:
                self.play(
                    Transform(current_net, new_net),
                    Transform(current_metrics_label, gen_label),
                    Transform(current_fitness_label, fitness_label),
                    run_time=1.5
                )
            
            current_net = new_net
            current_metrics_label = gen_label
            current_fitness_label = fitness_label
            last_topology = frame['topology']
            self.wait(1)

        final_msg = Text("Optimal Topology Converged", color=COLOR_CYAN).to_edge(DOWN)
        self.play(Write(final_msg))
        self.wait(2)

    def show_demo_text(self):
        txt = Text("No evolution_history.json found.\nRun NEAT in GUI and click Export.", color=RED)
        self.play(Write(txt))
        self.wait(2)

    def create_network_mobject(self, topology):
        """Creates a VGroup representing the network topology."""
        net = VGroup()
        
        nodes = topology['node_keys']
        conns = topology['connections']
        
        # Logic to separate types (simplified)
        inputs = [n for n in nodes if n < 0]
        outputs = [n for n in nodes if n >= 0 and n < 100]
        hidden = [n for n in nodes if n not in inputs and n not in outputs]
        
        pos = {}
        width = 10
        height = 6
        
        # Position nodes
        def assign_pos(node_list, x):
            count = len(node_list)
            for i, n in enumerate(node_list):
                y = (i / (count - 1 or 1)) * height - (height / 2)
                pos[n] = np.array([x, y, 0])

        assign_pos(inputs, -width/2 + 2)
        assign_pos(hidden, 0)
        assign_pos(outputs, width/2 - 2)
        
        # Connections first (so they are behind nodes)
        for (u, v), weight in conns:
            if u in pos and v in pos:
                color = COLOR_CYAN if weight > 0 else COLOR_ACCENT
                line = Line(pos[u], pos[v], color=color, stroke_width=abs(weight)*2)
                line.set_opacity(0.6)
                net.add(line)
        
        # Nodes
        for n in nodes:
            if n in pos:
                color = COLOR_CYAN if n in inputs else (COLOR_ACCENT if n in outputs else COLOR_HIDDEN)
                dot = Dot(pos[n], color=color, radius=0.15)
                # Add glow effect
                glow = Circle(radius=0.25, color=color, fill_opacity=0.2, stroke_width=0).move_to(dot)
                net.add(glow, dot)
                
        return net

# Entry point for command line: manim -ql video_exporter.py EvolutionScene
