---
phase: 3
plan: 3
wave: 2
---

# Plan 3.3: 3D Grid Search Plot

## Objective
Implémenter la recherche sur grille (Grid Search) pour explorer l'Espace de Recherche (Population vs Générations) et l'afficher sous forme de surface 3D dynamique.

## Context
- .gsd/SPEC.md (Goal 3: Grid Search 3D)
- ai_engine.py

## Tasks

<task type="auto">
  <name>Implement Grid Search Orchestrator</name>
  <files>grid_search.py</files>
  <action>
    Create grid_search.py with:
    1. `GridSearch` class:
        - Parameters: range for Population (X), range for Generations (Y).
        - Multi-processing support (REQ-04) using `multiprocessing.Pool` (12 workers/threads).
        - Optimization: reuse `AIEngine` for each point in the grid.
        - Results stored in a 2D numpy array (Z = best fitness).
    2. Progress reporting via callback to the GUI.
  </action>
  <verify>python -c "from grid_search import GridSearch; print('Import OK')"</verify>
  <done>grid_search.py can run a small 3x3 grid using multiple processes</done>
</task>

<task type="auto">
  <name>Create 3D Surface Plot component</name>
  <files>visualizer_3d.py, main_gui.py</files>
  <action>
    Create visualizer_3d.py with:
    1. 3D Plot using `dpg.add_plot` with 3D axes or a custom drawing.
    2. Alternatively use VisPy if DPG 3D is too limited (but VisPy might be harder to integrate on laptop). 
    3. Mapping the Z values to a color gradient (thermal style).
    4. Animation: Update the plot every 100 finished iterations (REQ-06).
  </action>
  <verify>python main_gui.py (Run Grid Search)</verify>
  <done>The 3D surface "mountain" grows in real-time as grid search progresses</done>
</task>

## Success Criteria
- [ ] Le grid search utilise les 12 threads du CPU (multi-processing).
- [ ] La surface 3D s'affiche et se met à jour dynamiquement.
- [ ] L'esthétique de la surface est harmonieuse avec le reste de l'application.
