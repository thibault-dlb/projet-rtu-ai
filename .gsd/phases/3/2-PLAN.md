---
phase: 3
plan: 2
wave: 1
---

# Plan 3.2: 2D Neural Network Visualizer

## Objective
Développer un composant de visualisation schématique du réseau de neurones (topology) qui s'adapte dynamiquement lors de l'évolution NEAT.

## Context
- neat_engine.py (Topology info: nodes, connections)
- main_gui.py

## Tasks

<task type="auto">
  <name>Create Drawing Component for Neural Nets</name>
  <files>visualizer_2d.py</files>
  <action>
    Create visualizer_2d.py with:
    1. Class `NetworkVisualizer`:
        - Uses `dpg.draw_layer` or `dpg.drawlist`.
        - Logic to position nodes in layers (Input -> Hidden -> Output).
        - `update_topology(topology_info)` method.
    2. Drawing logic:
        - Circles for nodes (colored by type: input/output/hidden).
        - Lines for connections.
        - Thickness variation based on weight magnitude (REQ-05).
        - Color variation based on weight sign (e.g., green for positive, Red for negative).
  </action>
  <verify>python -c "from visualizer_2d import NetworkVisualizer; print('Import OK')"</verify>
  <done>visualizer_2d.py exists and can draw a static network from mock topology info</done>
</task>

<task type="auto">
  <name>Integrate Visualizer into main UI</name>
  <files>main_gui.py, visualizer_2d.py</files>
  <action>
    1. Embed `NetworkVisualizer` into the central panel of `main_gui.py`.
    2. Connect the training callback to `visualizer.update_topology`.
    3. Ensure the drawing is performant even during fast generations.
  </action>
  <verify>python main_gui.py (Run NEAT and watch the network evolve)</verify>
  <done>NEAT evaluation shows the network topology changing in real-time on the UI</done>
</task>

## Success Criteria
- [ ] Le schéma du réseau s'affiche au centre de l'UI.
- [ ] L'épaisseur et la couleur des lignes reflètent les poids.
- [ ] La topologie se met à jour sans ralentir l'entraînement (60 FPS UI).
