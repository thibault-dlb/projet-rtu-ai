"""
Data loader for the Exoplanet Classification project.
Handles loading CSV files, splitting, and StandardScaler normalization.
"""
import os
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.config import (
    SEED, SPLIT_RATIO, DATA_DIR,
    KNOWN_DATA_FILE, UNKNOWN_DATA_FILE,
    LABEL_COLUMN, FEATURE_NAMES, LABEL_UNKNOWN,
)

# Named tuple for clean data access
DataBundle = namedtuple("DataBundle", [
    "X_train", "X_test", "y_train", "y_test",
    "X_unknown", "feature_names", "scaler",
])


def _set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data():
    """
    Load raw known_data and unknown_data CSV files.
    
    Returns:
        X_known (np.ndarray): Features of known objects (N, 12)
        y_known (np.ndarray): Labels of known objects (N,) — 0 or 1
        X_unknown (np.ndarray): Features of unknown objects (M, 12)
        feature_names (list): List of 12 feature column names
    """
    known_path = os.path.join(DATA_DIR, KNOWN_DATA_FILE)
    unknown_path = os.path.join(DATA_DIR, UNKNOWN_DATA_FILE)

    # Load with pandas
    df_known = pd.read_csv(known_path)
    df_unknown = pd.read_csv(unknown_path)

    # Separate features and labels
    feature_names = [col for col in FEATURE_NAMES if col in df_known.columns]
    
    X_known = df_known[feature_names].values.astype(np.float64)
    y_known = df_known[LABEL_COLUMN].values.astype(np.int32)
    X_unknown = df_unknown[feature_names].values.astype(np.float64)

    return X_known, y_known, X_unknown, feature_names


def split_data(X, y, split_ratio=SPLIT_RATIO, seed=SEED):
    """
    Split data into train and test sets with stratification.
    
    Args:
        X (np.ndarray): Feature matrix (N, D)
        y (np.ndarray): Label vector (N,)
        split_ratio (float): Fraction for training (default 0.8)
        seed (int): Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=split_ratio,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, X_unknown):
    """
    Normalize features using StandardScaler.
    Fit ONLY on training data to prevent data leakage.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        X_unknown (np.ndarray): Unknown features
    
    Returns:
        X_train_norm, X_test_norm, X_unknown_norm, scaler
    """
    scaler = StandardScaler()
    
    # Fit on train only
    X_train_norm = scaler.fit_transform(X_train)
    
    # Transform test and unknown with the same scaler
    X_test_norm = scaler.transform(X_test)
    X_unknown_norm = scaler.transform(X_unknown)

    return X_train_norm, X_test_norm, X_unknown_norm, scaler


def load_and_prepare_data(split_ratio=SPLIT_RATIO, seed=SEED):
    """
    Full data pipeline: load → split → normalize.
    
    Args:
        split_ratio (float): Train fraction (default 0.8)
        seed (int): Random seed (default 42)
    
    Returns:
        DataBundle: Named tuple with X_train, X_test, y_train, y_test,
                    X_unknown, feature_names, scaler
    """
    # Fix seeds
    _set_seeds(seed)

    # Load raw data
    X_known, y_known, X_unknown, feature_names = load_raw_data()

    # Print raw data summary
    n_planets = int(np.sum(y_known == 1))
    n_not_planets = int(np.sum(y_known == 0))
    print("=" * 55)
    print("  DONNÉES CHARGÉES")
    print("=" * 55)
    print(f"  Données connues  : {len(y_known):,} objets")
    print(f"    ├─ Planètes    : {n_planets:,} ({100*n_planets/len(y_known):.1f}%)")
    print(f"    └─ Non-planètes: {n_not_planets:,} ({100*n_not_planets/len(y_known):.1f}%)")
    print(f"  Données inconnues: {len(X_unknown):,} objets")
    print(f"  Features         : {len(feature_names)}")
    print("-" * 55)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X_known, y_known, split_ratio, seed
    )

    print(f"  Split {split_ratio:.0%} / {1-split_ratio:.0%} (seed={seed})")
    print(f"    ├─ Train : {len(y_train):,} ({int(np.sum(y_train==1)):,} planètes)")
    print(f"    └─ Test  : {len(y_test):,} ({int(np.sum(y_test==1)):,} planètes)")

    # Normalize
    X_train, X_test, X_unknown, scaler = normalize_data(
        X_train, X_test, X_unknown
    )

    print(f"  Normalisation    : StandardScaler (fit sur train)")
    print("=" * 55)

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_unknown=X_unknown,
        feature_names=feature_names,
        scaler=scaler,
    )


if __name__ == "__main__":
    # Quick test
    data = load_and_prepare_data()
    print(f"\nShapes:")
    print(f"  X_train : {data.X_train.shape}")
    print(f"  X_test  : {data.X_test.shape}")
    print(f"  y_train : {data.y_train.shape}")
    print(f"  y_test  : {data.y_test.shape}")
    print(f"  X_unknown: {data.X_unknown.shape}")
    print(f"  Features: {data.feature_names}")
