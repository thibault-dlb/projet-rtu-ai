# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
Créer une suite d'algorithmes d'intelligence artificielle de complexité croissante pour déterminer si un objet céleste observé par le télescope Kepler est une exoplanète ou non. Le projet progresse du plus simple (prédiction aléatoire) au plus avancé (NEAT et au-delà), permettant de comparer empiriquement les performances et les ressources nécessaires à chaque approche. Chaque algorithme produit un pourcentage de probabilité que l'objet soit une planète.

## Goals
1. Implémenter une progression d'algorithmes : Aléatoire → Hill Climbing → GA → NEAT → (HyperNEAT, WANN)
2. Entraîner et évaluer chaque algorithme sur les données connues (split 80/20)
3. Appliquer les modèles entraînés sur les données inconnues pour prédire leur classification
4. Visualiser en temps réel l'apprentissage de chaque algorithme (Pygame)
5. Produire des graphiques comparatifs détaillés des performances (matplotlib)
6. Mesurer et comparer les ressources nécessaires à l'entraînement de chaque algorithme

## Non-Goals (Out of Scope)
- Interface web ou API REST
- Déploiement en production
- Collecte ou nettoyage des données (datasets considérés propres)
- Algorithmes de deep learning classiques (CNN, RNN, etc.) — le focus est sur les approches évolutionnaires
- Optimisation des hyperparamètres automatisée (grid search, etc.)

## Users
Développeur/chercheur souhaitant comparer des approches algorithmiques pour la classification d'exoplanètes Kepler. Utilisation en local via la ligne de commande.

## Algorithms

### 1. Aléatoire (Baseline)
- Génère un pourcentage aléatoire pour chaque objet, sans aucune logique ni prise en compte des paramètres
- Sert uniquement de point de comparaison (baseline)

### 2. Hill Climbing
- Optimise les poids d'un perceptron simple par perturbations successives
- À chaque itération, perturbe légèrement les poids et conserve la modification si elle améliore le score
- Prend en compte les 12 features d'entrée

### 3. Algorithme Génétique Standard (GA)
- Population de solutions (vecteurs de poids pour un réseau de neurones simple)
- Sélection, croisement, mutation
- Évolution sur plusieurs générations jusqu'à stabilisation

### 4. NEAT (NeuroEvolution of Augmenting Topologies)
- Évolution simultanée de la topologie et des poids du réseau
- Complexification progressive du réseau
- Librairie : `neat-python`

### 5. HyperNEAT (Optionnel)
- Extension de NEAT utilisant les CPPNs pour encoder les patterns de connectivité
- Permet des réseaux plus grands et plus structurés

### 6. WANN — Weight Agnostic Neural Networks (Optionnel)
- Recherche de topologies performantes indépendamment des poids
- Un seul poids partagé pour tout le réseau

## Data

### Datasets
- **known_data** (~7328 lignes) : Objets célestes classifiés
  - `koi_disposition` = 1 → Planète confirmée
  - `koi_disposition` = 0 → Non-planète (faux positif)
- **unknown_data** (~1877 lignes) : Objets non classifiés
  - `koi_disposition` = 2 → À prédire

### Features (12 colonnes)
| Feature | Description |
|---------|-------------|
| `koi_period` | Période orbitale (jours) |
| `koi_impact` | Paramètre d'impact |
| `koi_duration` | Durée du transit (heures) |
| `koi_depth` | Profondeur du transit (ppm) |
| `koi_prad` | Rayon planétaire (rayon terrestre) |
| `koi_teq` | Température d'équilibre (K) |
| `koi_insol` | Flux d'insolation (flux terrestre) |
| `koi_model_snr` | SNR du modèle de transit |
| `koi_steff` | Température effective de l'étoile (K) |
| `koi_slogg` | Log de la gravité de surface stellaire |
| `koi_srad` | Rayon stellaire (rayon solaire) |
| `koi_smass` | Masse stellaire (masse solaire) |

### Split
- 80% entraînement / 20% test (sur known_data)
- Seed fixe pour reproductibilité

## Architecture

```
projet-rtu-ai/
├── datasets/
│   ├── known_data
│   └── unknown_data
├── shared/                    # Code commun
│   ├── __init__.py
│   ├── data_loader.py         # Chargement et split des données
│   ├── metrics.py             # Calcul de toutes les métriques
│   ├── visualization.py       # Graphiques matplotlib comparatifs
│   └── config.py              # Configuration globale (seed, split ratio, etc.)
├── algorithms/
│   ├── 01_random/             # Baseline aléatoire
│   │   └── main.py
│   ├── 02_hill_climbing/      # Hill Climbing sur perceptron
│   │   └── main.py
│   ├── 03_genetic_algorithm/  # GA standard
│   │   └── main.py
│   ├── 04_neat/               # NEAT
│   │   ├── main.py
│   │   └── config-neat.txt
│   ├── 05_hyperneat/          # HyperNEAT (optionnel)
│   │   └── main.py
│   └── 06_wann/               # WANN (optionnel)
│       └── main.py
├── results/                   # Résultats générés
│   ├── predictions/           # Prédictions sur unknown_data
│   ├── metrics/               # Métriques par algorithme
│   └── plots/                 # Graphiques comparatifs
└── requirements.txt
```

## Visualisation

### Temps réel (Pygame)
- **Hill Climbing** : Courbe de fitness en temps réel, poids du perceptron
- **GA** : Évolution de la population, meilleur/moyen/pire fitness par génération
- **NEAT** : Réseau de neurones du meilleur individu, complexification en direct
- **HyperNEAT/WANN** : Topologie du réseau en temps réel

### Résultats finaux (matplotlib)
- Comparaison des accuracy, precision, recall, F1-score entre algos
- Matrices de confusion pour chaque algo
- Courbes ROC/AUC
- Temps d'entraînement et ressources (CPU, mémoire)
- Distribution des prédictions sur les données inconnues

## Constraints
- **Langage** : Python 3.x
- **Indépendance** : Chaque algorithme exécutable indépendamment (`python algorithms/XX_name/main.py`)
- **Reproductibilité** : Seed fixe pour toutes les opérations aléatoires
- **Stabilisation** : Entraînement jusqu'à convergence (pas d'amélioration significative sur N générations/itérations)

## Metrics à calculer
- Accuracy (taux de bonne classification)
- Precision (par classe)
- Recall (par classe)
- F1-Score (par classe et macro/micro/weighted)
- Matrice de confusion
- Courbe ROC et AUC
- Log-loss
- Temps d'entraînement
- Utilisation mémoire RAM
- Nombre d'itérations/générations jusqu'à convergence

## Success Criteria
- [ ] Les 4 algorithmes principaux (Aléatoire, Hill Climbing, GA, NEAT) sont implémentés et fonctionnels
- [ ] Chaque algorithme produit un pourcentage de probabilité par objet
- [ ] Les visualisations Pygame temps réel fonctionnent pour chaque algorithme
- [ ] Les métriques comparatives sont calculées et affichées via matplotlib
- [ ] Les prédictions sur les données inconnues sont exportées en CSV
- [ ] Chaque algorithme est exécutable de manière indépendante
- [ ] Les résultats montrent une progression claire des performances du Random vers NEAT
- [ ] HyperNEAT et/ou WANN sont implémentés (bonus)
