---
phase: 8
plan: 1
wave: 1
---

# Plan 8.1: Dashboard Global — Backend API & Frontend Setup

## Objective
Mettre en place la fondation de l'interface unifiée. Initialiser une application Next.js premium et un serveur Backend FastAPI qui servira de pont pour contrôler et monitorer les algorithmes d'IA.

## Context
- .gsd/SPEC.md
- .gsd/DECISIONS.md
- shared/config.py (pour les paramètres des algos)

## Tasks

<task type="auto">
  <name>Initialiser le Backend FastAPI (dashboard/backend/)</name>
  <files>dashboard/backend/main.py, dashboard/backend/api.py, requirements_dashboard.txt</files>
  <action>
    Créer un serveur backend léger pour :
    1. Lister les algorithmes disponibles et leurs hyperparamètres par défaut (via `shared.config`).
    2. Endpoint pour lancer un algorithme avec paramètres personnalisés (utilisant `subprocess` ou `threading` pour ne pas bloquer l'API).
    3. Système de WebSockets pour streamer les logs et les données de progression (fitness, itération actuelle) vers le frontend.
  </action>
  <verify>Lancement manuel et test via Swagger UI (/docs)</verify>
  <done>API capable de lister les algos et de lancer une exécution factice avec retour temps réel.</done>
</task>

<task type="auto">
  <name>Initialiser le Frontend Next.js (dashboard/frontend/)</name>
  <files>dashboard/frontend/</files>
  <action>
    Initialiser une application Next.js 14+ avec :
    1. Tailwind CSS pour le stylage.
    2. Lucide-react pour les icônes.
    3. Framer Motion pour les animations "WOW" (transitions de pages, menus fluides).
    4. Créer le layout principal : Sidebar (navigation algos) et zone de contenu principale (Dashboard Global).
    5. Implémenter un thème sombre premium (Glassmorphism, dégradés subtils).
  </action>
  <verify>npm run dev sur le frontend</verify>
  <done>Frontend accessible, design premium visible, navigation de base fonctionnelle.</done>
</task>

<task type="auto">
  <name>Créer le Dashboard Global</name>
  <files>dashboard/frontend/app/page.tsx</files>
  <action>
    Implémenter la page d'accueil du dashboard montrant :
    1. Un résumé des performances de tous les algos (lu depuis `results/summary.md` via le backend).
    2. Des graphiques de comparaison rapides (réutilisation des images matplotlib ou recréation en Recharts).
    3. État actuel du projet (phases complétées).
  </action>
  <verify>Visualisation des données du projet sur la home page</verify>
  <done>Dashboard global affichant les derniers résultats connus.</done>
</task>

## Success Criteria
- [ ] Le backend peut piloter un script Python et récolter ses sorties.
- [ ] Le frontend est esthétiquement réussi (WOW factor).
- [ ] La communication temps réel (WebSocket) est établie.
- [ ] L'utilisateur peut naviguer entre le dashboard et les pages spécifiques de chaque IA.
