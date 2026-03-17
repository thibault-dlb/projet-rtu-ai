---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: DPG Foundation & Layout

## Objective
Mettre en place la structure de base de l'application Dear PyGui avec une esthétique "Engineering/Gaming". Création des panneaux de contrôle et de la boucle principale avec gestion du multi-threading pour éviter les freezes.

## Context
- .gsd/SPEC.md (Esthétique Opera GX, 60 FPS)
- ai_engine.py (Unified AI interface)

## Tasks

<task type="auto">
  <name>Setup DPG window and "Opera GX" theme</name>
  <files>main_gui.py</files>
  <action>
    Create main_gui.py with:
    1. DPG initialization (viewport creation, setup).
    2. Theme definition:
        - Dark background (near black).
        - Accent colors: Secondary red/orange (Opera GX style), cyan for highlights.
        - Rounded corners and sleek borders.
    3. Main layout using `dpg.add_window`:
        - Sidebar (Left): Control Panel.
        - Central Area: Visualizers (Placeholder for now).
        - Bottom: Metrics & Threshold slider.
  </action>
  <verify>python main_gui.py</verify>
  <done>GUI window opens with a dark theme and defined sections</done>
</task>

<task type="auto">
  <name>Implement Control Panel and Multi-threading orchestration</name>
  <files>main_gui.py</files>
  <action>
    Enhance main_gui.py with:
    1. Control widgets:
        - Algorithm selector (Random, HC, GA, NEAT).
        - Pop size, Iterations inputs.
        - Start/Stop buttons.
    2. Logic Threading:
        - Function `run_training_thread` to run `AIEngine.train` in a separate `threading.Thread`.
        - Thread-safe callback to update DPG widgets from the training thread.
        - Loading indicator or progress bar.
    3. Basic Metrics Display:
        - Real-time display of Best Fitness, F1, and iteration count.
  </action>
  <verify>python main_gui.py</verify>
  <done>Buttons trigger a background training thread that updates metrics on the UI without freezing</done>
</task>

## Success Criteria
- [ ] L'application se lance et respecte l'esthétique dark/gaming.
- [ ] Le thread de calcul ne bloque pas l'UI.
- [ ] Les métriques se mettent à jour en direct durant l'entraînement.
