# 🪐 Exoplanet AI Suite — Evolutionary Classifier

> **Native Windows Application for Exoplanet Discovery using NEAT & Genetic Algorithms.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Dear PyGui](https://img.shields.io/badge/UI-Dear%20PyGui-red.svg)](https://github.com/hoffstadt/DearPyGui)
[![Manim](https://img.shields.io/badge/Render-Manim-cyan.svg)](https://www.manim.community/)

---

## 🔭 Vision
Ce projet est une suite d'intelligence artificielle haute performance conçue pour classifier les objets célestes (Kepler Objects of Interest) en tant que **Planètes** ou **Autres**. Il utilise des algorithmes évolutionnaires pour optimiser non seulement les paramètres mais aussi la topologie des réseaux de neurones (NEAT).

L'interface, inspirée de l'esthétique "Opera GX", offre une expérience fluide à **60 FPS** avec des visualisations 2D et 3D en temps réel.

## 🚀 Fonctionnalités Clés
- **IA Évolutionnaire Unifiée** : Support pour NEAT (NeuroEvolution of Augmenting Topologies), Algorithmes Génétiques (GA) et Hill Climbing.
- **Visualisation 2D Live** : Observez la structure du réseau de neurones évoluer dynamiquement à chaque génération.
- **Grid Search 3D** : Exploration multi-processus de l'espace des hyperparamètres (Population x Générations) avec rendu de surface 3D.
- **Reporting Premium (Manim)** : Exportation automatique de vidéos haute fidélité illustrant la croissance topologique du meilleur génome.
- **Dashboard d'Inférence** : Classification massive de plus de 1700 objets inconnus avec export CSV des résultats.

## 🛠 Stack Technique
- **Core** : Python 3.12, NumPy, Pandas, Scikit-Learn.
- **IA** : `neat-python`, PyTorch (Inférence).
- **GUI** : Dear PyGui (Accélération GPU).
- **Visualisation** : VisPy (Isometric Projections), Matplotlib.
- **Vidéo** : Manim (Animation Engine).

## 📥 Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/thibault-dlb/projet-rtu-ai.git
   cd projet-rtu-ai
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   *Note : Manim nécessite l'installation de FFmpeg sur votre système.*

## 🎮 Utilisation

Lancez l'interface principale :
```bash
python main_gui.py
```

1. **Entraînement** : Choisissez un algorithme (ex: NEAT), réglez la population et cliquez sur `START TRAINING`.
2. **Optimisation** : Lancez un `RUN GRID SEARCH (3D)` pour voir la montagne de performance se construire.
3. **Rapport** : Une fois satisfait, cliquez sur `EXPORT VIDEO REPORT` pour générer une animation Manim.
4. **Classification Finale** : Allez dans l'onglet `Final Reports` pour traiter les données `unknown_data.csv`.

## 📊 Performance
Optimisé pour :
- **CPU** : Ryzen 5 5500 (12 threads simulés pour le Grid Search).
- **GPU** : Compatible RTX (via DPG/PyTorch).

---
*Développé dans le cadre du projet RTU-AI.*
