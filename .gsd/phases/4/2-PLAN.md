---
phase: 4
plan: 2
wave: 1
---

# Plan 4.2: Final Analytics & Unknown Prediction

## Objective
Finaliser le pipeline de données en classifiant les 1700+ échantillons inconnus et en générant un rapport de synthèse final (CSV/Stats).

## Context
- data_loader.py (unknown_data.csv)
- ai_engine.py (best_model inference)

## Tasks

<task type="auto">
  <name>Implement Prediction System for Unknown Objects</name>
  <files>inference_system.py</files>
  <action>
    Create inference_system.py to:
    1. Load the "best model" saved in Phase 2/3.
    2. Process `unknown_data.csv` through `ExoplanetDataLoader.load_unknown()`.
    3. Generate probability scores and binary classes.
    4. Save results to `classification_results.csv`.
  </action>
  <verify>python inference_system.py</verify>
  <done>A CSV file is generated with 1700+ rows and classification labels</done>
</task>

<task type="auto">
  <name>Final Analysis Dashboard (Tab in GUI)</name>
  <files>main_gui.py</files>
  <action>
    Add a "Final Reports" tab in the GUI to:
    1. Show a summary of predictions (e.g., Pie chart of Planets vs Others in unknown data).
    2. Display the F1/Precision/Recall matrix of the best model.
    3. Allow batch-exporting all results.
  </action>
  <verify>python main_gui.py (View Dashboard)</verify>
  <done>The GUI provides a summary of findings on the unknown population</done>
</task>

## Success Criteria
- [ ] Prédiction effectuée sur 100% du dataset `unknown_data.csv`.
- [ ] Export CSV propre incluant les IDs et les probabilités.
- [ ] Tableau de bord final affichant les statistiques clés.
