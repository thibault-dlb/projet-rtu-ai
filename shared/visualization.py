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
        if file.endswith("_metrics.json") and not file.startswith("."):
            algo_id = file.replace("_metrics.json", "")
            try:
                with open(os.path.join(METRICS_DIR, file), 'r') as f:
                    all_metrics[algo_id] = json.load(f)
            except Exception as e:
                print(f"  [Error] Failed to load {file}: {e}")
    return all_metrics

def _load_all_resources():
    """Load all resource stats from METRICS_DIR."""
    all_resources = {}
    if not os.path.exists(METRICS_DIR):
        return all_resources
        
    for file in os.listdir(METRICS_DIR):
        if file.endswith("_resources.json"):
            algo_id = file.replace("_resources.json", "")
            try:
                with open(os.path.join(METRICS_DIR, file), 'r') as f:
                    all_resources[algo_id] = json.load(f)
            except Exception as e:
                print(f"  [Error] Failed to load {file}: {e}")
    return all_resources

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

def plot_resource_comparison():
    """Create plots for time and memory comparison."""
    res_data = _load_all_resources()
    if not res_data:
        return

    algos = list(res_data.keys())
    # Sort for consistent display
    algos.sort()
    
    display_names = [ALGO_DISPLAY_NAMES.get(a, a) for a in algos]
    times = [res_data[a].get('elapsed_time_sec', 0) for a in algos]
    mems = [res_data[a].get('peak_memory_mb', 0) for a in algos]
    colors = [ALGO_COLORS.get(a, '#95a5a6') for a in algos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Time plot
    ax1.barh(display_names, times, color=colors)
    ax1.set_title("Temps d'Entraînement (secondes)")
    ax1.set_xlabel("Secondes")
    for i, v in enumerate(times):
        ax1.text(v + 0.1, i, f"{v:.2f}s", va='center')

    # Memory plot
    ax2.barh(display_names, mems, color=colors)
    ax2.set_title("Utilisation Mémoire Pic (Mo)")
    ax2.set_xlabel("Mo")
    for i, v in enumerate(mems):
        ax2.text(v + 0.1, i, f"{v:.1f}MB", va='center')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "resource_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Comparaison ressources sauvegardée dans {save_path}")

def plot_roc_curves():
    """Superimpose ROC curves of all algorithms."""
    metrics_data = _load_all_metrics()
    if not metrics_data:
        return

    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Baseline 50%
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Aléatoire (Théorique)")

    for algo, data in metrics_data.items():
        if "roc_curve" in data:
            fpr = data["roc_curve"]["fpr"]
            tpr = data["roc_curve"]["tpr"]
            auc = data.get("roc_auc", 0)
            plt.plot(fpr, tpr, label=f"{ALGO_DISPLAY_NAMES.get(algo, algo)} (AUC={auc:.3f})",
                     color=ALGO_COLORS.get(algo), linewidth=2)

    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC', fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Courbes ROC sauvegardées dans {save_path}")

def generate_all_plots():
    """Generate all baseline plots."""
    print("\nGénération des graphiques comparatifs...")
    plot_metrics_comparison()
    plot_resource_comparison()
    plot_roc_curves()
    print("Graphiques générés.")

if __name__ == "__main__":
    generate_all_plots()
