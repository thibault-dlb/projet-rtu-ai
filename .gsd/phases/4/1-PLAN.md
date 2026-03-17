---
phase: 4
plan: 1
wave: 1
---

# Plan 4.1: Manim Evolution Render

## Objective
Générer une vidéo automatisée montrant l'évolution topologique du meilleur génome trouvé par NEAT. Cette vidéo sert de rapport visuel "premium" pour comprendre comment l'IA a structuré ses décisions.

## Context
- neat_engine.py (History and Genome topology)
- SPEC.md (Goal 4: Manim export)

## Tasks

<task type="auto">
  <name>Implement Manim Exporter mapping</name>
  <files>video_exporter.py</files>
  <action>
    Create video_exporter.py with:
    1. `EvolutionScene` class inheriting from Manim `Scene`.
    2. Logic to convert `neat_engine` topology history into a sequence of Manim animations.
    3. Visual style: High-contrast, neon-glow (Opera GX style), nodes morphing and connections appearing with weights.
  </action>
  <verify>python -m manim video_exporter.py EvolutionScene -ql</verify>
  <done>A video file is generated showing the best neural network growing over generations</done>
</task>

<task type="auto">
  <name>Integrate Export button in GUI</name>
  <files>main_gui.py, video_exporter.py</files>
  <action>
    1. Add "EXPORT VIDEO REPORT" button in `main_gui.py`.
    2. Trigger the Manim render process (via subprocess or direct call) and show a progress bar/status.
  </action>
  <verify>Launch GUI, run NEAT, click Export Video</verify>
  <done>The user can trigger a 10s video export of their best model result from the UI</done>
</task>

## Success Criteria
- [ ] Production d'un fichier .mp4 montrant l'évolution.
- [ ] Fluidité des transitions entre les blocs topologiques.
- [ ] Esthétique cohérente avec la charte graphique rouge/cyan.
