"""
inference_system.py
Generates predictions on unknown exoplanet data using a trained engine.
Saves results to classification_results.csv.
"""

import pandas as pd
import numpy as np
import os
from ai_engine import AIEngine

class InferenceSystem:
    def __init__(self, engine: AIEngine, loader):
        self.engine = engine
        self.loader = loader

    def run_inference(self, output_path="classification_results.csv"):
        """Run classification on unknown_data and save to CSV."""
        X_unknown = self.loader.load_unknown()
        if X_unknown is None:
            return None

        # Predict using default threshold 0.5
        y_pred = self.engine.predict(X_unknown, threshold=0.5)
        
        # We need IDs to make the result useful. 
        # Re-loading unknown_data to get the original columns (e.g. kepid)
        df_raw = pd.read_csv(self.loader.unknown_path)
        
        # Filter raw df if loader dropped any rows (though unlikely for unknown)
        # For simplicity, we assume row alignment is preserved (no NaNs dropped in loader)
        # If NaNs were dropped, we'd need to align.
        
        results_df = pd.DataFrame({
            'kepid': df_raw['kepid'],
            'prediction': y_pred,
            'label': ['Planet' if p == 1 else 'Other' for p in y_pred]
        })
        
        results_df.to_csv(output_path, index=False)
        return results_df

    def get_stats(self, results_df):
        """Compute basic stats on the final classifications."""
        counts = results_df['label'].value_counts()
        total = len(results_df)
        planet_pct = counts.get('Planet', 0) / total * 100
        return {
            'total': total,
            'planets': counts.get('Planet', 0),
            'others': counts.get('Other', 0),
            'planet_pct': planet_pct
        }
