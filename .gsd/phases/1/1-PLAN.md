---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Configuration et Data Loader

## Objective
Créer les modules fondamentaux `config.py` et `data_loader.py` qui seront utilisés par tous les algorithmes. Le data loader doit charger les CSV, normaliser avec StandardScaler, et splitter 80/20.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md
- datasets/known_data
- datasets/unknown_data

## Tasks

<task type="auto">
  <name>Créer shared/config.py</name>
  <files>shared/config.py</files>
  <action>
    Créer le fichier de configuration avec les constantes globales :
    - SEED = 42
    - SPLIT_RATIO = 0.8
    - DEFAULT_THRESHOLD = 0.5
    - DATA_DIR = chemin vers datasets/ (relatif à la racine du projet)
    - RESULTS_DIR = chemin vers results/
    - KNOWN_DATA_FILE = "known_data"
    - UNKNOWN_DATA_FILE = "unknown_data"
    - FEATURE_NAMES = liste des 12 noms de colonnes
    - LABEL_COLUMN = "koi_disposition"
    - Utiliser os.path pour construire les chemins de manière cross-platform
    - Ajouter un PROJECT_ROOT calculé dynamiquement via os.path
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); from shared.config import *; print(f'SEED={SEED}, SPLIT={SPLIT_RATIO}, FEATURES={len(FEATURE_NAMES)}')"</verify>
  <done>config.py importable sans erreur, toutes les constantes accessibles</done>
</task>

<task type="auto">
  <name>Créer shared/__init__.py</name>
  <files>shared/__init__.py</files>
  <action>
    Créer un __init__.py qui expose les modules principaux pour faciliter les imports.
    Import de config, data_loader (quand disponible), metrics (quand disponible).
    Pour l'instant, importer seulement config.
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); import shared; print('shared package OK')"</verify>
  <done>Le package shared est importable</done>
</task>

<task type="auto">
  <name>Créer shared/data_loader.py</name>
  <files>shared/data_loader.py</files>
  <action>
    Implémenter le module de chargement des données :
    
    1. Fonction `load_raw_data()` :
       - Charger known_data et unknown_data avec pandas (read_csv)
       - Séparer features (X) et labels (y) pour known_data
       - Retourner en numpy arrays
    
    2. Fonction `split_data(X, y, split_ratio, seed)` :
       - Utiliser sklearn.model_selection.train_test_split
       - Stratified split (conserver la proportion planète/non-planète)
       - Seed fixe pour reproductibilité
    
    3. Fonction `normalize_data(X_train, X_test, X_unknown)` :
       - Créer un StandardScaler
       - FIT uniquement sur X_train (pas de data leakage)
       - TRANSFORM sur X_train, X_test, et X_unknown
       - Retourner les 3 arrays normalisés + le scaler
    
    4. Fonction principale `load_and_prepare_data()` :
       - Appelle les 3 fonctions ci-dessus dans l'ordre
       - Retourne un dict/namedtuple avec :
         X_train, X_test, y_train, y_test, X_unknown, feature_names, scaler
       - Imprime un résumé des données (tailles, proportion de classes)
    
    IMPORTANT : 
    - Ne PAS utiliser les labels des données inconnues (colonne = 2)
    - Fixer numpy.random.seed ET random.seed au début
    - Les feature_names doivent rester accessibles pour les graphiques
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); from shared.data_loader import load_and_prepare_data; data = load_and_prepare_data(); print(f'Train: {data.X_train.shape}, Test: {data.X_test.shape}, Unknown: {data.X_unknown.shape}')"</verify>
  <done>Data loader retourne les données normalisées avec les bonnes dimensions : train ~5862 samples, test ~1466 samples, unknown ~1877 samples, 12 features chacun</done>
</task>

## Success Criteria
- [ ] `shared/config.py` contient toutes les constantes nécessaires
- [ ] `shared/data_loader.py` charge, split, et normalise les données correctement
- [ ] StandardScaler est fit UNIQUEMENT sur les données d'entraînement
- [ ] Le split est stratifié et reproductible (seed fixe)
- [ ] Les dimensions sont correctes (~5862 train, ~1466 test, 1877 unknown, 12 features)
