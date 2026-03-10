"""
Shared configuration for the Exoplanet Classification project.
All global constants and paths are defined here.
"""
import os

# ─── Project Root ───────────────────────────────────────────────
# Dynamically resolve project root (parent of shared/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Reproducibility ───────────────────────────────────────────
SEED = 42

# ─── Data Split ────────────────────────────────────────────────
SPLIT_RATIO = 0.8  # 80% train, 20% test

# ─── Classification Threshold ──────────────────────────────────
DEFAULT_THRESHOLD = 0.5

# ─── Paths ─────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# ─── Data Files ────────────────────────────────────────────────
KNOWN_DATA_FILE = "known_data"
UNKNOWN_DATA_FILE = "unknown_data"

# ─── Label Column ──────────────────────────────────────────────
LABEL_COLUMN = "koi_disposition"

# ─── Label Values ──────────────────────────────────────────────
LABEL_PLANET = 1
LABEL_NOT_PLANET = 0
LABEL_UNKNOWN = 2

# ─── Feature Names (12 Kepler features) ───────────────────────
FEATURE_NAMES = [
    "koi_period",      # Période orbitale (jours)
    "koi_impact",      # Paramètre d'impact
    "koi_duration",    # Durée du transit (heures)
    "koi_depth",       # Profondeur du transit (ppm)
    "koi_prad",        # Rayon planétaire (rayon terrestre)
    "koi_teq",         # Température d'équilibre (K)
    "koi_insol",       # Flux d'insolation (flux terrestre)
    "koi_model_snr",   # SNR du modèle de transit
    "koi_steff",       # Température effective de l'étoile (K)
    "koi_slogg",       # Log gravité de surface stellaire
    "koi_srad",        # Rayon stellaire (rayon solaire)
    "koi_smass",       # Masse stellaire (masse solaire)
]

# ─── Algorithm Names ───────────────────────────────────────────
ALGO_NAMES = [
    "01_random",
    "02_hill_climbing",
    "03_genetic_algorithm",
    "04_neat",
    "05_hyperneat",
    "06_wann",
]

# ─── Algorithm Display Names ──────────────────────────────────
ALGO_DISPLAY_NAMES = {
    "01_random": "Aléatoire",
    "02_hill_climbing": "Hill Climbing",
    "03_genetic_algorithm": "Algorithme Génétique",
    "04_neat": "NEAT",
    "05_hyperneat": "HyperNEAT",
    "06_wann": "WANN",
}

# ─── Colors for plots (one per algorithm) ─────────────────────
ALGO_COLORS = {
    "01_random": "#95a5a6",          # Gris — baseline
    "02_hill_climbing": "#3498db",   # Bleu
    "03_genetic_algorithm": "#2ecc71",  # Vert
    "04_neat": "#e74c3c",            # Rouge
    "05_hyperneat": "#9b59b6",       # Violet
    "06_wann": "#f39c12",            # Orange
}

# ─── Algorithm Hyperparameters (Defaults) ──────────────────────
# Ces valeurs sont utilisées par défaut mais peuvent être surchargées
# par les arguments de la ligne de commande.
ALGO_HYPERPARAMS = {
    "02_hill_climbing": {
        "iterations": 5000,
        "perturb_strength": 0.05
    },
    "03_genetic_algorithm": {
        "generations": 100,
        "pop_size": 60,
        "mutation_rate": 0.1
    },
    "04_neat": {
        "generations": 50,
        "pop_size": 50 # Note: pop_size est aussi dans config-neat.txt
    }
}
