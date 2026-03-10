from scipy.sparse._construct import _random
import os
import sys
import pygame
import numpy as np
import threading
import time

from shared.ui import Button, Slider, SidebarItem, COLORS
from shared.config import (
    ALGO_DISPLAY_NAMES, ALGO_NAMES, RESULTS_DIR, 
    ALGO_HYPERPARAMS, SEED
)
from shared.data_loader import load_and_prepare_data

import importlib.util

def import_runner(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Dynamically import Runners
HillClimbingRunner = import_runner("HillClimbingRunner", "algorithms/02_hill_climbing/main.py").HillClimbingRunner
GARunner = import_runner("GARunner", "algorithms/03_genetic_algorithm/main.py").GARunner
NEATRunner = import_runner("NEATRunner", "algorithms/04_neat/main.py").NEATRunner

# Simple Random Runner
class RandomRunner:
    def __init__(self, data):
        self.data = data
        self.running = False

    def start(self): 
        # On utilise l'import dynamique car le dossier commence par un chiffre
        mod = import_runner("RandomMain", "algorithms/01_random/main.py")
        mod.main() # Just run once
        self.running = False

    def step(self):
        return False

    def draw(self, surface): 
        surface.fill((10, 10, 15))
        f = pygame.font.SysFont("Arial", 24)
        surface.blit(f.render("Baseline Aléatoire - Exécution terminée", True, (255, 255, 255)), (100, 100))

    def finish(self):
        return {}, 0.5

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Antigravity AI - Dashboard Kepler")
clock = pygame.time.Clock()
font_main = pygame.font.SysFont("Arial", 28, bold=True)
font_sub = pygame.font.SysFont("Arial", 18)
font_mono = pygame.font.SysFont("Consolas", 14)

class DashboardApp:
    def __init__(self):
        self.running = True
        self.active_tab = "Synthèse Globale"
        self.sidebar_width = 250
        
        # UI State
        self.current_runner = None
        self.is_training = False
        self.data = None
        
        # Sidebar
        self.sidebar_items = [SidebarItem(0, 100, self.sidebar_width, 50, "Synthèse Globale")]
        self.sidebar_items[0].selected = True
        for i, name in enumerate(ALGO_NAMES):
            self.sidebar_items.append(SidebarItem(0, 150 + i * 50, self.sidebar_width, 50, ALGO_DISPLAY_NAMES.get(name, name)))

        # Params for current algo
        self.sliders = []
        self.play_btn = Button(250 + 50, HEIGHT - 100, 200, 50, "LANCER", color=COLORS["accent"])
        
        self.summary_content = ""
        self.load_summary()
        
        # Load Data in background thread
        threading.Thread(target=self.lazy_load_data, daemon=True).start()

    def lazy_load_data(self):
        np.random.seed(SEED)
        self.data = load_and_prepare_data()

    def load_summary(self):
        summary_path = os.path.join(RESULTS_DIR, "summary.md")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    self.summary_content = f.read()
            except:
                with open(summary_path, "r", encoding="latin-1") as f:
                    self.summary_content = f.read()

    def setup_algo_params(self, algo_display_name):
        self.sliders = []
        # Find internal name
        internal_name = next((k for k, v in ALGO_DISPLAY_NAMES.items() if v == algo_display_name), None)
        if not internal_name or internal_name not in ALGO_HYPERPARAMS: return
        
        hp = ALGO_HYPERPARAMS[internal_name]
        y_off = 150
        for key, val in hp.items():
            # Heuristic for min/max
            min_val, max_val = 1, val * 10
            if "strength" in key or "rate" in key:
                min_val, max_val = 0.01, 1.0
            
            s = Slider(self.sidebar_width + 40, y_off, 220, 10, min_val, max_val, val, label=key, is_int=("strength" not in key and "rate" not in key))
            self.sliders.append(s)
            y_off += 70

    def start_training(self):
        if not self.data: return
        
        internal_name = next((k for k, v in ALGO_DISPLAY_NAMES.items() if v == self.active_tab), None)
        params = {s.label: s.value for s in self.sliders}
        
        if internal_name == "01_random":
            self.current_runner = RandomRunner(self.data)
        elif internal_name == "02_hill_climbing":
            self.current_runner = HillClimbingRunner(self.data, **params)
        elif internal_name == "03_genetic_algorithm":
            self.current_runner = GARunner(self.data, **params)
        elif internal_name == "04_neat":
            self.current_runner = NEATRunner(self.data, **params)
            
        if self.current_runner:
            self.current_runner.start()
            self.is_training = True

    def run(self):
        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_down = pygame.mouse.get_pressed()[0]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Sidebar clicks
                    for item in self.sidebar_items:
                        if item.rect.collidepoint(mouse_pos):
                            for other in self.sidebar_items: other.selected = False
                            item.selected = True
                            self.active_tab = item.label
                            self.is_training = False
                            self.current_runner = None
                            self.setup_algo_params(self.active_tab)
                    
                    # Play button
                    if self.play_btn.rect.collidepoint(mouse_pos) and self.active_tab != "Synthèse Globale" and not self.is_training:
                        self.start_training()

            # Logic
            if self.is_training and self.current_runner:
                still_running = self.current_runner.step()
                if not still_running:
                    self.current_runner.finish()
                    self.is_training = False
                    self.load_summary() # Refresh summary

            # Visuals
            screen.fill(COLORS["bg"])
            
            # Sidebar
            pygame.draw.rect(screen, COLORS["sidebar"], (0, 0, self.sidebar_width, HEIGHT))
            logo = font_main.render("ANTIGRAVITY", True, COLORS["accent"])
            screen.blit(logo, (20, 30))
            for item in self.sidebar_items: item.update(mouse_pos, mouse_down); item.draw(screen)

            # Content Area
            rect = pygame.Rect(self.sidebar_width + 30, 30, WIDTH - self.sidebar_width - 60, HEIGHT - 60)
            
            if self.active_tab == "Synthèse Globale":
                self.draw_overview(rect)
            else:
                self.draw_algo_view(rect)

            pygame.display.flip()
            clock.tick(60)

    def draw_overview(self, rect):
        title = font_main.render("Synthèse des Performances", True, (255, 255, 255))
        screen.blit(title, (rect.x, rect.y))
        
        box = pygame.Rect(rect.x, rect.y + 60, rect.w, rect.h - 80)
        pygame.draw.rect(screen, COLORS["panel"], box, border_radius=15)
        
        lines = self.summary_content.split("\n")
        yy = box.y + 20
        for line in lines[:28]:
            color = (200, 200, 220)
            if "|" in line: color = COLORS["accent"]
            t = font_mono.render(line, True, color)
            screen.blit(t, (box.x + 20, yy))
            yy += 20

    def draw_algo_view(self, rect):
        title = font_main.render(self.active_tab, True, (255, 255, 255))
        screen.blit(title, (rect.x, rect.y))
        
        # Params Panel
        p_rect = pygame.Rect(rect.x, rect.y + 60, 300, rect.h - 80)
        pygame.draw.rect(screen, COLORS["panel"], p_rect, border_radius=15)
        
        if not self.is_training:
            for s in self.sliders: s.update(pygame.mouse.get_pos(), pygame.mouse.get_pressed()[0]); s.draw(screen)
            self.play_btn.update(pygame.mouse.get_pos(), pygame.mouse.get_pressed()[0])
            self.play_btn.draw(screen)
        else:
            msg = font_sub.render("Entraînement en cours...", True, COLORS["secondary"])
            screen.blit(msg, (p_rect.x + 20, p_rect.y + 20))

        # Viz Area
        v_rect = pygame.Rect(p_rect.right + 20, rect.y + 60, rect.w - 320, rect.h - 80)
        v_surf = pygame.Surface((v_rect.w, v_rect.h))
        if self.current_runner:
            self.current_runner.draw(v_surf)
        else:
            v_surf.fill((5, 5, 10))
            m = font_sub.render("En attente de lancement", True, (80, 80, 100))
            v_surf.blit(m, (v_rect.w//2 - m.get_width()//2, v_rect.h//2))
            
        screen.blit(v_surf, v_rect.topleft)
        pygame.draw.rect(screen, (50, 50, 70), (v_rect.x, v_rect.y, v_rect.w, v_rect.h), 1, border_radius=15)

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
    pygame.quit()
