---
phase: 1
plan: 2
wave: 2
---

# Plan 1.2: Métriques et Mesure de Ressources

## Objective
Créer le module `metrics.py` pour calculer toutes les métriques de performance, et un module `resource_monitor.py` pour mesurer automatiquement le temps et la mémoire de chaque algorithme. Inclure aussi le calcul du seuil optimal.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md (ADR-11, ADR-12)
- shared/config.py (créé dans Plan 1.1)

## Tasks

<task type="auto">
  <name>Créer shared/metrics.py</name>
  <files>shared/metrics.py</files>
  <action>
    Implémenter le module de calcul de métriques :
    
    1. Fonction `compute_all_metrics(y_true, y_pred_proba, threshold=0.5)` :
       - Convertir y_pred_proba en y_pred_binary avec le seuil donné
       - Calculer :
         * Accuracy
         * Precision (par classe et weighted)
         * Recall (par classe et weighted)
         * F1-Score (par classe, macro, micro, weighted)
         * Matrice de confusion
         * ROC AUC
         * Log-loss
       - Retourner un dict structuré avec toutes les métriques
    
    2. Fonction `find_optimal_threshold(y_true, y_pred_proba)` :
       - Calculer la courbe ROC (fpr, tpr, thresholds) avec sklearn
       - Trouver le seuil optimal via Youden's J statistic : max(tpr - fpr)
       - Retourner le seuil optimal et le J-statistic correspondant
    
    3. Fonction `save_metrics(metrics_dict, algo_name, results_dir)` :
       - Sauvegarder les métriques dans un JSON dans results/metrics/{algo_name}_metrics.json
       - Inclure : algo_name, threshold utilisé, seuil optimal, toutes les métriques
    
    4. Fonction `save_predictions(y_pred_proba, algo_name, results_dir)` :
       - Sauvegarder les prédictions sur unknown_data dans results/predictions/{algo_name}_predictions.csv
       - Colonnes : index, probability, prediction (binaire avec seuil optimal)
    
    5. Fonction `print_metrics_summary(metrics_dict, algo_name)` :
       - Afficher un résumé formaté dans le terminal
       - Inclure les métriques principales + seuil optimal
    
    IMPORTANT :
    - Gérer le cas où y_pred_proba est constant (ex: algo random) → ROC AUC peut planter
    - Utiliser sklearn.metrics pour tous les calculs
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); from shared.metrics import compute_all_metrics, find_optimal_threshold; import numpy as np; y_true = np.array([1,0,1,0,1]); y_proba = np.array([0.9,0.1,0.8,0.3,0.7]); m = compute_all_metrics(y_true, y_proba); print(f'Accuracy={m[\"accuracy\"]:.2f}, AUC={m[\"roc_auc\"]:.2f}'); t, j = find_optimal_threshold(y_true, y_proba); print(f'Optimal threshold={t:.2f}')"</verify>
  <done>Toutes les métriques sont calculées correctement, seuil optimal trouvé, sauvegarde fonctionnelle</done>
</task>

<task type="auto">
  <name>Créer shared/resource_monitor.py</name>
  <files>shared/resource_monitor.py</files>
  <action>
    Implémenter un context manager et un decorator pour mesurer les ressources :
    
    1. Classe `ResourceMonitor` (context manager) :
       - Mesurer le temps d'exécution (time.perf_counter)
       - Mesurer l'utilisation mémoire RAM (tracemalloc ou psutil)
       - Enregistrer le nombre d'itérations/générations (passé en paramètre après exécution)
       - Méthode `get_stats()` retournant un dict avec :
         * elapsed_time (secondes)
         * peak_memory_mb (Mo)
         * iterations (si défini)
    
    2. Fonction `save_resource_stats(stats, algo_name, results_dir)` :
       - Sauvegarder dans results/metrics/{algo_name}_resources.json
    
    Usage prévu :
    ```python
    with ResourceMonitor() as monitor:
        # ... entraînement de l'algo ...
        monitor.iterations = num_iterations
    stats = monitor.get_stats()
    save_resource_stats(stats, "hill_climbing", RESULTS_DIR)
    ```
    
    Utiliser tracemalloc (stdlib) plutôt que psutil pour éviter une dépendance supplémentaire.
  </action>
  <verify>python -c "import sys; sys.path.insert(0, '.'); from shared.resource_monitor import ResourceMonitor; import time; rm = ResourceMonitor(); rm.__enter__(); time.sleep(0.1); rm.__exit__(None,None,None); s = rm.get_stats(); print(f'Time={s[\"elapsed_time\"]:.2f}s, Memory={s[\"peak_memory_mb\"]:.1f}MB')"</verify>
  <done>ResourceMonitor mesure correctement temps et mémoire, stats sauvegardables en JSON</done>
</task>

## Success Criteria
- [ ] `compute_all_metrics()` retourne accuracy, precision, recall, F1, confusion matrix, ROC AUC, log-loss
- [ ] `find_optimal_threshold()` retourne le seuil Youden's J correct
- [ ] `save_metrics()` et `save_predictions()` écrivent des fichiers lisibles
- [ ] `ResourceMonitor` mesure temps et mémoire correctement
- [ ] Gestion des edge cases (prédictions constantes, etc.)
