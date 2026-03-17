"""
data_loader.py
Loads and preprocesses the Kepler exoplanet datasets.

Target column: koi_disposition (1 = Planet, 0 = Other)
Features (12): koi_period, koi_impact, koi_duration, koi_depth,
               koi_prad, koi_teq, koi_insol, koi_model_snr,
               koi_steff, koi_slogg, koi_srad, koi_smass
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


FEATURES = [
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass'
]
TARGET = 'koi_disposition'


class ExoplanetDataLoader:
    """Handles loading, cleaning, and normalizing the Kepler datasets."""

    def __init__(self, known_path='known_data.csv', unknown_path='unknown_data.csv',
                 test_size=0.2, random_state=42):
        self.known_path = known_path
        self.unknown_path = unknown_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()

    # ------------------------------------------------------------------
    def load_known(self):
        """Load known_data.csv, clean, scale, and split into train/test."""
        df = pd.read_csv(self.known_path)

        # Drop rows with any NaN in the feature columns or target
        df = df.dropna(subset=FEATURES + [TARGET])

        X = df[FEATURES].values
        y = df[TARGET].values.astype(int)

        # Fit scaler on the full known dataset, then transform
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    def load_unknown(self):
        """Load unknown_data.csv, clean, and scale using the fitted scaler."""
        if not os.path.exists(self.unknown_path):
            print(f"Warning: {self.unknown_path} not found.")
            return None

        df = pd.read_csv(self.unknown_path)
        df = df.dropna(subset=FEATURES)
        X = df[FEATURES].values
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    def summary(self, X_train, X_test, y_train, y_test):
        """Print a quick summary of the loaded data."""
        total = len(y_train) + len(y_test)
        planets = int(y_train.sum() + y_test.sum())
        others = total - planets
        print("=" * 50)
        print("  EXOPLANET DATA SUMMARY")
        print("=" * 50)
        print(f"  Total samples : {total}")
        print(f"  Train / Test  : {len(y_train)} / {len(y_test)}")
        print(f"  Planets (1)   : {planets}  ({100*planets/total:.1f}%)")
        print(f"  Others  (0)   : {others}  ({100*others/total:.1f}%)")
        print(f"  Features      : {len(FEATURES)}")
        print(f"  X range       : [{X_train.min():.4f}, {X_train.max():.4f}]")
        print("=" * 50)


# ======================================================================
if __name__ == "__main__":
    loader = ExoplanetDataLoader()
    X_train, X_test, y_train, y_test = loader.load_known()
    loader.summary(X_train, X_test, y_train, y_test)
