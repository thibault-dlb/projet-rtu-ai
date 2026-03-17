# REQUIREMENTS.md

## Functional Requirements
| ID | Requirement | Source | Status |
|----|-------------|--------|--------|
| REQ-01 | Prétraitement des données (MinMaxScaler) sur 12 features | SPEC Goal 2 | Pending |
| REQ-02 | Implémentation de NEAT pour la classification binaire (Planet/Other) | SPEC Goal 2 | Pending |
| REQ-03 | Interface native Dear PyGui avec 60 FPS | SPEC Goal 1 | Pending |
| REQ-04 | Architecture multi-threadée (UI vs Logic vs Compute) | SPEC Goal 5 | Pending |
| REQ-05 | Visualisation dynamique de la topologie du réseau de neurones | SPEC Goal 4 | Pending |
| REQ-06 | Grid Search 3D (Population x Générations) | SPEC Goal 3 | Pending |
| REQ-07 | Export de vidéo Manim de l'évolution topologique | SPEC Goal 4 | Pending |
| REQ-08 | Matrice de confusion et score F1 en temps réel | SPEC Goal 3 | Pending |

## Technical Requirements
- Python threading/multiprocessing pour la séparation des tâches.
- PyTorch avec support CUDA pour l'accélération GPU des modèles compatibles.
- Intégration VisPy ou DPG Native Drawing pour la 3D.
