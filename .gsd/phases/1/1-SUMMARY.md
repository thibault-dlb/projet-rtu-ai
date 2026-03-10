# Plan 1.1 Summary: Configuration et Data Loader

## Objective
Mettre en place les bases du projet : configuration globale et pipeline de données (chargement, split, normalisation).

## Changes
- Créé `shared/config.py` : centralisation des constantes (seed=42, split=0.8, chemins, noms de features).
- Créé `shared/data_loader.py` : chargement CSV, split stratifié, et `StandardScaler` (fit sur train uniquement).
- Créé `shared/__init__.py` : package shared opérationnel.

## Verification
- Commande : `python shared/data_loader.py`
- Résultat :
  - Train: (5860, 12)
  - Test: (1466, 12)
  - Unknown: (1875, 12)
  - Pas de data leakage détecté (scaler fit on train).

## Verdict
**PASS**
