"""
Visualization module for the Exoplanet Classification project.
Generates comparative plots (matplotlib) and interactive threshold analysis.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from shared.config import (
    ALGO_NAMES, ALGO_DISPLAY_NAMES, ALGO_COLORS,
    METRICS_DIR, PLOTS_DIR
)

def _load_all_metrics():
    """Load all metrics from METRICS_DIR."""
    all_metrics = {}
    if not os.path.exists(METRICS_DIR):
        return all_metrics
        
    for file in os.listdir(METRICS_DIR):
        if file.endswith("_metrics.json"):
            algo_id = file.replace("_metrics.json", "")
            try:
                with open(os.path.join(METRICS_DIR, file), 'r') as f:
                    all_metrics[algo_id] = json.load(f)
            except Exception as e:
                print(f"  [Error] Failed to load {file}: {e}")
    return all_metrics

def plot_metrics_comparison():
    """Create grouped barplot for performance metrics comparison."""
    metrics_data = _load_all_metrics()
    if not metrics_data:
        print("  [Warning] Aucune donnée de métrique trouvée.")
        return

    algos = list(metrics_data.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Précision", "Rappel", "F1-Score"]
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(algos)
    
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    for i, algo in enumerate(algos):
        values = [metrics_data[algo].get(m, 0) for m in metric_names]
        offset = (i - len(algos)/2 + 0.5) * width
        plt.bar(x + offset, values, width, 
                label=ALGO_DISPLAY_NAMES.get(algo, algo),
                color=ALGO_COLORS.get(algo, None))

    plt.xlabel('Métriques')
    plt.ylabel('Score')
    plt.title('Comparaison des Performances par Algorithme', fontsize=14)
    plt.xticks(x, display_names)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3)
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Comparaison sauvegardée dans {save_path}")

def interactive_threshold_viewer(y_true, y_pred_proba, algo_name):
    """
    Launch interactive window to explore threshold impact.
    Note: Requires a GUI environment.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.25)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Threshold slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Seuil', 0.0, 1.0, valinit=0.5)
    
    def update(val):
        threshold = slider.val
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Recalculate metrics
        cm = confusion_matrix(y_true, y_pred)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Update confusion matrix plot
        ax[0].clear()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Planète", "Planète"])
        disp.plot(ax=ax[0], cmap='Blues', colorbar=False)
        ax[0].set_title(f"Matrice de Confusion (Seuil={threshold:.2f})")
        
        # Update metrics barplot
        ax[1].clear()
        names = ['Accuracy', 'Précision', 'Rappel', 'F1']
        scores = [acc, prec, rec, f1]
        ax[1].bar(names, scores, color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'])
        ax[1].set_ylim(0, 1)
        ax[1].set_title("Métriques en Temps Réel")
        for i, v in enumerate(scores):
            ax[1].text(i, v + 0.02, f"{v:.3f}", ha='center')
            
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0.5)
    plt.suptitle(f"Analyse Interactive du Seuil - {ALGO_DISPLAY_NAMES.get(algo_name, algo_name)}")
    plt.show()

def generate_all_plots():
    """Generate all baseline plots."""
    print("\nGénération des graphiques comparatifs...")
    plot_metrics_comparison()
    # Add confusion matrix and ROC curves once data is available in later phases
    print("Graphiques générés.")

if __name__ == "__main__":
    generate_all_plots()
