"""
visualizer_2d.py
Neural Network Visualizer for Dear PyGui.
Draws nodes and weighted connections that evolve dynamically.
"""

import dearpygui.dearpygui as dpg
import numpy as np

class NetworkVisualizer:
    def __init__(self, parent_tag, width=800, height=500):
        self.parent = parent_tag
        self.width = width
        self.height = height
        self.node_radius = 12
        self.draw_list_tag = "net_draw_list"
        
        # Colors
        self.c_input = (0, 200, 255)    # Cyan-ish
        self.c_hidden = (200, 200, 200) # Grey
        self.c_output = (255, 60, 60)   # Opera GX Red
        self.c_pos = (50, 255, 50)      # Green
        self.c_neg = (255, 50, 50)      # Red
        
        with dpg.drawlist(width=width, height=height, parent=self.parent, tag=self.draw_list_tag):
            pass

    def clear(self):
        dpg.delete_item(self.draw_list_tag, children_only=True)

    def update_topology(self, topology_info):
        """
        Draws the network based on topology_info.
        topology_info: {
            'num_nodes': int,
            'node_keys': list,
            'connections': list of ((in_node, out_node), weight)
        }
        """
        self.clear()
        if not topology_info or 'node_keys' not in topology_info:
            return

        # Simple Layering Logic
        # Inputs: -1 to -N
        # Outputs: 0 to M
        # Hidden: rest
        
        nodes = topology_info['node_keys']
        conns = topology_info.get('connections', [])
        
        inputs = [n for n in nodes if n < 0]
        outputs = [n for n in nodes if n >= 0 and n < 100] # Simple heuristic for DPG/NEAT
        hidden = [n for n in nodes if n not in inputs and n not in outputs]
        
        # Position nodes
        positions = {}
        
        def set_pos(node_list, x_pct):
            count = len(node_list)
            for i, node in enumerate(node_list):
                y_pct = (i + 1) / (count + 1)
                positions[node] = (x_pct * self.width, y_pct * self.height)

        set_pos(inputs, 0.1)
        set_pos(hidden, 0.5)
        set_pos(outputs, 0.9)
        
        # Draw Connections
        for (u, v), weight in conns:
            if u in positions and v in positions:
                p1 = positions[u]
                p2 = positions[v]
                
                color = self.c_pos if weight > 0 else self.c_neg
                thickness = min(6, abs(weight) * 2)
                
                dpg.draw_line(p1, p2, color=color, thickness=thickness, parent=self.draw_list_tag)
        
        # Draw Nodes
        for node in nodes:
            if node in positions:
                pos = positions[node]
                color = self.c_input if node < 0 else (self.c_output if node in outputs else self.c_hidden)
                
                dpg.draw_circle(pos, self.node_radius, color=(255, 255, 255), fill=color, parent=self.draw_list_tag)
                # dpg.draw_text((pos[0]-5, pos[1]-5), str(node), size=12, color=(0,0,0), parent=self.draw_list_tag)
