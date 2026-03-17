"""
visualizer_3d.py
3D Surface Plotter for Dear PyGui.
Uses a simple wireframe projection to visualize Grid Search results.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import math

class SurfacePlot3D:
    def __init__(self, parent_tag, width=400, height=400):
        self.parent = parent_tag
        self.width = width
        self.height = height
        self.draw_list_tag = "surface_3d_draw_list"
        
        # Camera / Projection params
        self.angle_x = -0.5
        self.angle_z = 0.5
        self.scale = 200
        self.offset = (width // 2, height // 1.5)
        
        with dpg.drawlist(width=width, height=height, parent=self.parent, tag=self.draw_list_tag):
            pass

    def clear(self):
        dpg.delete_item(self.draw_list_tag, children_only=True)

    def project(self, x, y, z):
        """Simple isometric/perspective projection."""
        # Rotate around Z
        nx = x * math.cos(self.angle_z) - y * math.sin(self.angle_z)
        ny = x * math.sin(self.angle_z) + y * math.cos(self.angle_z)
        
        # Rotate around X
        nz = z * math.cos(self.angle_x) - ny * math.sin(self.angle_x)
        ny = z * math.sin(self.angle_x) + ny * math.cos(self.angle_x)
        
        px = nx * self.scale + self.offset[0]
        py = -nz * self.scale + self.offset[1] # Negative Z is up in projection
        return (px, py)

    def update_surface(self, x_vals, y_vals, z_matrix):
        """
        x_vals (1D), y_vals (1D), z_matrix (2D: shape len(y), len(x))
        """
        self.clear()
        if z_matrix is None or z_matrix.size == 0:
            return

        ny, nx = z_matrix.shape
        
        # Normalize coordinates to [-1, 1] for projection
        def norm_x(i): return (i / (nx-1) * 2 - 1) if nx > 1 else 0
        def norm_y(j): return (j / (ny-1) * 2 - 1) if ny > 1 else 0
        
        # Draw Mesh Lines
        for j in range(ny):
            for i in range(nx):
                # Line to next in X
                if i < nx - 1:
                    p1 = self.project(norm_x(i), norm_y(j), z_matrix[j, i])
                    p2 = self.project(norm_x(i+1), norm_y(j), z_matrix[j, i+1])
                    color = self.get_color(z_matrix[j, i])
                    dpg.draw_line(p1, p2, color=color, thickness=1, parent=self.draw_list_tag)
                
                # Line to next in Y
                if j < ny - 1:
                    p1 = self.project(norm_x(i), norm_y(j), z_matrix[j, i])
                    p2 = self.project(norm_x(i), norm_y(j+1), z_matrix[j+1, i])
                    color = self.get_color(z_matrix[j, i])
                    dpg.draw_line(p1, p2, color=color, thickness=1, parent=self.draw_list_tag)

    def get_color(self, val):
        """Map value [0, 1] to a thermal-like gradient (cyan to red)."""
        r = int(255 * val)
        g = int(255 * (1 - abs(val - 0.5) * 2))
        b = int(255 * (1 - val))
        return (r, g, b, 200)
