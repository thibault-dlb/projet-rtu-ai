"""
main_gui.py
Main GUI for the Exoplanet Classification System using Dear PyGui.
Implements an 'Opera GX' inspired theme and a multi-threaded architecture.
"""

import dearpygui.dearpygui as dpg
import threading
import time
import numpy as np
import subprocess
import os
from ai_engine import AIEngine, AlgorithmType
from data_loader import ExoplanetDataLoader
from visualizer_2d import NetworkVisualizer
from visualizer_3d import SurfacePlot3D
from grid_search import GridSearch
from inference_system import InferenceSystem

# --- CONFIGURATION & THEME ---
COLORS = {
    "bg": (15, 15, 15),
    "sidebar": (25, 25, 25),
    "accent": (255, 60, 60),  # Opera GX Red
    "accent_hover": (255, 100, 100),
    "text": (240, 240, 240),
    "highlight": (0, 255, 255),  # Cyan
}

class ExoplanetGUI:
    def __init__(self):
        self.loader = ExoplanetDataLoader()
        self.X_train, self.X_test, self.y_train, self.y_test = (np.array([]), np.array([]), np.array([]), np.array([]))
        self.engine = None
        self.visualizer = None
        self.visualizer_3d = None
        self.training_thread = threading.Thread()
        self.grid_thread = threading.Thread()
        self.inference_results = None
        self.is_training = False
        
        # UI State Variables
        self.status_text = "Ready"
        self.current_f1 = 0.0
        self.current_gen = 0
        
        self.setup_dpg()
        self.apply_theme()
        self.create_layout()
        self.load_initial_data()

    def setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='Exoplanet AI Suite - Evolutionary Classifier', width=1200, height=800)
        dpg.setup_dearpygui()

    def apply_theme(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLORS["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLORS["sidebar"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, COLORS["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS["accent_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS["text"])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
        dpg.bind_theme(global_theme)

    def load_initial_data(self):
        self.update_status("Loading data...")
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = self.loader.load_known()
            self.update_status(f"Data Loaded: {len(self.y_train) + len(self.y_test)} samples")
        except Exception as e:
            self.update_status(f"Data Error: {e}")

    def update_status(self, text):
        self.status_text = text
        if dpg.does_item_exist("status_label"):
            dpg.set_value("status_label", f"Status: {text}")

    def on_start_training(self):
        if self.is_training: return
        
        algo_name = dpg.get_value("algo_combo")
        algo_type = AlgorithmType[algo_name]
        pop_size = dpg.get_value("pop_size_input")
        gens = dpg.get_value("gens_input")
        
        self.engine = AIEngine(algo_type, pop_size=pop_size, generations=gens, light_mode=True)
        self.is_training = True
        self.update_status(f"Training {algo_name}...")
        
        self.training_thread = threading.Thread(target=self.train_worker, daemon=True)
        self.training_thread.start()

    def train_worker(self):
        def callback(gen, metrics):
            self.current_gen = gen
            self.current_f1 = metrics['best_fitness']
            
            # Update visualizer if topology in metrics
            if 'topology' in metrics and self.visualizer:
                self.visualizer.update_topology(metrics['topology'])
            
            # Thread-safe UI update
            dpg.set_value("metric_gen", f"Gen: {gen}")
            dpg.set_value("metric_f1", f"F1 (best): {self.current_f1:.4f}")
            
        try:
            res = self.engine.train(self.X_train, self.y_train, self.X_test, self.y_test, callback=callback)
            
            # Save history for video export if NEAT
            if self.engine.algorithm_type == AlgorithmType.NEAT:
                self.engine.save_history()
                
            self.update_status(f"Training Complete. Test F1: {res.test_f1:.4f}")
        except Exception as e:
            self.update_status(f"Train Error: {e}")
        finally:
            self.is_training = False

    def on_start_grid_search(self):
        if self.is_training: return
        
        algo_name = dpg.get_value("algo_combo")
        algo_type = AlgorithmType[algo_name]
        
        self.is_training = True
        self.update_status("Grid Search Running...")
        
        def grid_worker():
            try:
                gs = GridSearch(algo_type, n_workers=12)
                
                def update_cb(curr, total, point):
                    dpg.set_value("status_label", f"Grid: {curr}/{total} | {point[0]}p, {point[1]}g")
                
                pop_range = [10, 20, 50, 100, 200]
                gen_range = [10, 20, 50, 100]
                
                res = gs.run(pop_range, gen_range, callback=update_cb)
                
                # Update 3D plot
                if self.visualizer_3d:
                    self.visualizer_3d.update_surface(res['x'], res['y'], res['z'])
                
                self.update_status(f"Grid Complete in {res['time']:.1f}s")
            except Exception as e:
                self.update_status(f"Grid Error: {e}")
            finally:
                self.is_training = False

        self.grid_thread = threading.Thread(target=grid_worker, daemon=True)
        self.grid_thread.start()

    def on_export_video(self):
        if self.is_training: return
        
        if not os.path.exists("evolution_history.json"):
            self.update_status("No history found. Run NEAT training first.")
            return

        self.is_training = True
        self.update_status("Exporting Video (Manim)... Please wait.")
        
        def export_worker():
            try:
                # Run manim in low quality for speed
                cmd = ["python", "-m", "manim", "video_exporter.py", "EvolutionScene", "-ql", "--progress_bar", "none"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.update_status("Video Exported Successfully! (Check media/videos)")
                else:
                    self.update_status(f"Export Failed: {result.stderr[:100]}")
            except Exception as e:
                self.update_status(f"Export Error: {e}")
            finally:
                self.is_training = False

        threading.Thread(target=export_worker, daemon=True).start()

    def on_run_inference(self):
        if self.engine is None or self.engine.best_model is None:
            self.update_status("No trained model. Train NEAT or GA first.")
            return
            
        self.update_status("Running Inference on Unknown Data...")
        inf = InferenceSystem(self.engine, self.loader)
        results = inf.run_inference()
        
        if results is not None:
            self.inference_results = inf.get_stats(results)
            self.update_status(f"Inference Complete. Found {self.inference_results['planets']} candidates.")
            
            # Update Pie Chart
            pts = [self.inference_results['planets'], self.inference_results['others']]
            dpg.set_value("pie_series", [pts])
            dpg.set_value("inf_total_text", f"Total: {self.inference_results['total']}")
            dpg.set_value("inf_planet_text", f"Planets: {self.inference_results['planets']}")

    def create_layout(self):
        with dpg.window(label="Main", tag="PrimaryWindow"):
            with dpg.group(horizontal=True):
                # Sidebar
                with dpg.child_window(width=300, border=True):
                    dpg.add_text("CONTROL PANEL", color=COLORS["accent"])
                    dpg.add_separator()
                    dpg.add_spacer(height=10)
                    
                    dpg.add_text("Algorithm:")
                    dpg.add_combo(list(AlgorithmType.__members__.keys()), default_value="GA", tag="algo_combo", width=-1)
                    
                    dpg.add_spacer(height=5)
                    dpg.add_text("Population Size:")
                    dpg.add_input_int(default_value=50, tag="pop_size_input", width=-1)
                    
                    dpg.add_spacer(height=5)
                    dpg.add_text("Generations:")
                    dpg.add_input_int(default_value=50, tag="gens_input", width=-1)
                    
                    dpg.add_spacer(height=20)
                    dpg.add_button(label="START TRAINING", callback=self.on_start_training, width=-1, height=40)
                    dpg.add_button(label="RUN GRID SEARCH (3D)", callback=self.on_start_grid_search, width=-1, height=40)
                    dpg.add_button(label="EXPORT VIDEO REPORT", callback=self.on_export_video, width=-1, height=40)
                    dpg.add_button(label="STOP", width=-1)
                    
                    dpg.add_spacer(height=20)
                    dpg.add_text("Status:", color=COLORS["highlight"])
                    dpg.add_text("Status: Ready", tag="status_label", wrap=280)

                # Main Content
                with dpg.group():
                    with dpg.tab_bar():
                        with dpg.tab(label="Neural Network (2D)"):
                            with dpg.child_window(height=500, border=True, tag="viz_window"):
                                dpg.add_text("NEURAL NETWORK TOPOLOGY", color=COLORS["accent"])
                                dpg.add_separator()
                                self.visualizer = NetworkVisualizer("viz_window", width=860, height=450)
                        
                        with dpg.tab(label="Performance Surface (3D)"):
                            with dpg.child_window(height=500, border=True, tag="viz_3d_window"):
                                dpg.add_text("GRID SEARCH SURFACE", color=COLORS["accent"])
                                dpg.add_separator()
                                self.visualizer_3d = SurfacePlot3D("viz_3d_window", width=860, height=450)
                        
                        with dpg.tab(label="Final Reports"):
                            with dpg.child_window(height=500, border=True):
                                dpg.add_text("INVESTIGATION SUMMARY", color=COLORS["accent"])
                                dpg.add_separator()
                                
                                with dpg.group(horizontal=True):
                                    with dpg.child_window(width=300, height=400, border=True):
                                        dpg.add_text("Action:")
                                        dpg.add_button(label="RUN CLASSIFICATION", callback=self.on_run_inference, width=-1, height=40)
                                        dpg.add_spacer(height=20)
                                        dpg.add_text("Stats:")
                                        dpg.add_text("Total: --", tag="inf_total_text")
                                        dpg.add_text("Planets: --", tag="inf_planet_text")
                                        dpg.add_spacer(height=20)
                                        dpg.add_button(label="OPEN CSV RESULTS", width=-1)

                                    with dpg.plot(label="Candidate Distribution", width=-1, height=-1):
                                        dpg.add_plot_legend()
                                        with dpg.plot_axis(dpg.mvXAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True):
                                            pass
                                        with dpg.plot_axis(dpg.mvYAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True):
                                            dpg.add_pie_series(0.5, 0.5, 0.4, [0.1, 0.9], ["Planets", "Others"], tag="pie_series")
                    
                    # Bottom Panel: Metrics
                    with dpg.child_window(height=-1, border=True):
                        dpg.add_text("LIVE METRICS", color=COLORS["accent"])
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Gen: 0", tag="metric_gen")
                                dpg.add_text("F1 (best): 0.0000", tag="metric_f1")
                            dpg.add_spacer(width=50)
                            with dpg.group():
                                dpg.add_text("Decision Threshold:")
                                dpg.add_slider_float(default_value=0.5, min_value=0.0, max_value=1.0, width=400)

        dpg.set_primary_window("PrimaryWindow", True)

    def run(self):
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    app = ExoplanetGUI()
    app.run()
