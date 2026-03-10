"""
Comparison Script for Exoplanet Classification project.
Loads results from all algorithms and generates comparative visualizations.
"""
import os
import sys
import pandas as pd

# Set path to root for shared module access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.visualization import generate_all_plots, _load_all_metrics
from shared.config import RESULTS_DIR, ALGO_DISPLAY_NAMES

def generate_summary_table(metrics_data):
    """Generate a markdown summary table of results."""
    if not metrics_data:
        return "Aucune donnée disponible."
        
    rows = []
    for algo_id, m in metrics_data.items():
        rows.append({
            "Algorithme": ALGO_DISPLAY_NAMES.get(algo_id, algo_id),
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "F1-Score": f"{m.get('f1', 0):.4f}",
            "ROC AUC": f"{m.get('roc_auc', 0):.4f}",
            "Log-Loss": f"{m.get('log_loss', 0):.4f}"
        })
        
    df = pd.DataFrame(rows)
    # Sort by accuracy
    df = df.sort_values("Accuracy", ascending=False)
    return df.to_markdown(index=False)

def main():
    print("=" * 60)
    print("  SYNTHÈSE DES RÉSULTATS")
    print("=" * 60)
    
    # 1. Generate Plots
    generate_all_plots()
    
    # 2. Generate Markdown Summary
    metrics_data = _load_all_metrics()
    table = generate_summary_table(metrics_data)
    
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Rapport de Synthèse — Classification d'Exoplanètes\n\n")
        f.write("Ce rapport compare les performances des différents algorithmes d'IA implémentés dans le cadre du projet.\n\n")
        f.write("## Performances Comparées\n\n")
        f.write(table)
        f.write("\n\n*Note: Les graphiques détaillés sont disponibles dans le dossier `results/plots/`.*")
        
    print("\nTableau Récapitulatif :")
    print(table)
    print(f"\nRapport complet sauvegardé dans {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
