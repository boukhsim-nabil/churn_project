# RetainIQ v2.0 — Plateforme IA de Prédiction & Rétention du Churn

> Projet Industriel 2024-2025 · XGBoost · Streamlit · APScheduler · SHAP · SendGrid · Gmail SMTP · SQLite · bcrypt

---

## Table des matières

- [Positionnement & Vision Produit](#positionnement--vision-produit)
1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Nouveautés de la v2.0](#2-nouveautés-de-la-v20)
3. [Architecture générale](#3-architecture-générale)
4. [Stack technique](#4-stack-technique)
5. [Structure des fichiers](#5-structure-des-fichiers)
6. [Schéma de la base de données](#6-schéma-de-la-base-de-données)
7. [Modèle de Données — Data Model](#7-modèle-de-données--data-model)
8. [Installation et démarrage](#8-installation-et-démarrage)
9. [Variables d'environnement](#9-variables-denvironnement)
10. [Workflows Fonctionnels Détaillés](#10-workflows-fonctionnels-détaillés)
    - 10.1 [Workflow 1 — Ingestion et Prédiction ML](#101-workflow-1--ingestion-et-prédiction-ml)
    - 10.2 [Workflow 2 — Moteur de Triage Statistique](#102-workflow-2--moteur-de-triage-statistique)
    - 10.3 [Workflow 3 — Action en Boucle Fermée](#103-workflow-3--action-en-boucle-fermée)
    - 10.4 [Workflow 4 — Automatisation et Gouvernance](#104-workflow-4--automatisation-et-gouvernance)
11. [Workflow complet — étape par étape (v2.0)](#11-workflow-complet--étape-par-étape-v20)
    - 11.1 [Authentification et inscription](#111-authentification-et-inscription)
    - 11.2 [Blank Slate — premier démarrage](#112-blank-slate--premier-démarrage)
    - 11.3 [Import CSV et pipeline de données](#113-import-csv-et-pipeline-de-données)
    - 11.4 [Entraînement du modèle XGBoost](#114-entraînement-du-modèle-xgboost)
    - 11.5 [Dashboard Overview et Visual Analytics](#115-dashboard-overview-et-visual-analytics)
    - 11.6 [Prédiction IA et jauge de risque](#116-prédiction-ia-et-jauge-de-risque)
    - 11.7 [Simulateur What-If](#117-simulateur-what-if)
    - 11.8 [Alertes Clients](#118-alertes-clients)
    - 11.9 [Explainabilité SHAP](#119-explainabilité-shap)
    - 11.10 [Assistant IA (Chatbot)](#1110-assistant-ia-chatbot)
    - 11.11 [Programme de Fidélité (NOUVEAU v2.0)](#1111-programme-de-fidélité-nouveau-v20)
    - 11.12 [Rapports planifiés et scheduler](#1112-rapports-planifiés-et-scheduler)
12. [Documentation complète des modules](#12-documentation-complète-des-modules)
    - [churn_prediction_dashboard.py](#churn_prediction_dashboardpy)
    - [auth.py](#authpy)
    - [database.py](#databasepy)
    - [data_pipeline.py](#data_pipelinepy)
    - [shap_explainer.py](#shap_explainerpy)
    - [email_reports.py](#email_reportspy)
    - [weekly_report_job.py](#weekly_report_jobpy)
    - [scheduler.py](#schedulerpy)
    - [loyalty_page.py](#loyalty_pagepy)
    - [loyalty_config.py](#loyalty_configpy)
    - [loyalty_messages_job.py](#loyalty_messages_jobpy)
13. [Dictionnaire des Fonctions Core](#13-dictionnaire-des-fonctions-core)
14. [Pages du dashboard (toutes)](#14-pages-du-dashboard-toutes)
15. [Système de segmentation clients v2.0](#15-système-de-segmentation-clients-v20)
16. [Catalogue de récompenses](#16-catalogue-de-récompenses)
17. [Scheduler — deux jobs automatiques](#17-scheduler--deux-jobs-automatiques)
18. [Sécurité et Garde-Fous](#18-sécurité-et-garde-fous)
19. [Limites connues et pistes d'amélioration](#19-limites-connues-et-pistes-damélioration)

---

## Positionnement & Vision Produit

### De l'IA Prédictive à l'IA Prescriptive

RetainIQ s'inscrit dans une trajectoire d'évolution fondamentale : la plateforme ne se limite pas à *prédire* le churn — elle orchestre la **réponse opérationnelle complète**, formant une boucle de rétention autonome et mesurable.

```
┌─────────────────────────────────────────────────────────────┐
│              BOUCLE DE RÉTENTION AUTONOME                   │
│                                                             │
│   DÉTECTER       COMPRENDRE       AGIR          MESURER     │
│   ────────       ───────────      ────          ────────    │
│  XGBoost ML  →  SHAP + Triage  → Campagne  →  KPIs + PDF   │
│  ChurnProba     Motif de Risque   Fidélité     Scheduler    │
└─────────────────────────────────────────────────────────────┘
```

### SaaS B2B Agnostique

Contrairement aux solutions sectorielles rigides, RetainIQ adopte une architecture **agnostique au domaine métier** :

- **Détection automatique des colonnes** : l'outil s'adapte à la structure du CSV fourni, sans configuration manuelle obligatoire, grâce à un moteur d'inférence basé sur des heuristiques nommées et des fallbacks binaires.
- **Moteur de triage statistique universel** : les seuils de risque sont calculés dynamiquement à partir des distributions réelles du jeu de données (quantile P75), rendant les règles métier indépendantes de toute terminologie sectorielle.
- **5 secteurs préconfigurés** : Télécom, Fitness, E-commerce, EdTech, SaaS B2B — chacun avec ses colonnes attendues, ses récompenses et ses messages de gratitude.

### Architecture API-First

RetainIQ est conçu pour s'intégrer nativement à tout écosystème CRM/ERP via une interface **API-First** basée sur des *payloads* JSON standardisés. Chaque déclenchement de campagne génère un objet structuré transmissible par webhook :

```json
{
  "event":     "loyalty_campaign_triggered",
  "timestamp": "2025-05-01T10:32:00",
  "company":   "NomEntreprise",
  "secteur":   "📱 Télécom",
  "reward": {
    "id": 3, "label": "Cadeau Ancienneté",
    "action": "Offrir", "cible": "Clients 12+ mois",
    "valeur": "1 mois gratuit", "duree": "30 jours"
  },
  "targeting": {
    "priority_filter": "🔴 Critique (>80%)",
    "motif_filter":    "Pression tarifaire",
    "total_clients":   12
  },
  "clients_sample": [
    {
      "churn_proba": 0.87,
      "priorite":    "🔴 Critique (>80%)",
      "motif": "Pression tarifaire"
    }
  ]
}
```

Ce *payload* standardisé permet une intégration directe avec Salesforce, HubSpot, Pipedrive ou tout système HTTP/REST sans développement intermédiaire.

---

## 1. Vue d'ensemble du projet

**RetainIQ** est une plateforme web full-stack de prédiction et de rétention du churn client basée sur l'intelligence artificielle. Elle permet à des entreprises de cinq secteurs d'activité de :

- **Détecter** les clients susceptibles de partir avant qu'ils ne le fassent (modèle XGBoost personnalisé par utilisateur)
- **Comprendre** les raisons du risque grâce à l'explainabilité SHAP
- **Simuler** l'impact d'une action commerciale avant de la déclencher (What-If Simulator)
- **Alerter** en temps réel sur les clients à risque élevé
- **Fidéliser** les clients stables grâce à un programme de récompenses automatisé (NOUVEAU v2.0)
- **Automatiser** l'envoi de rapports PDF hebdomadaires et de messages de gratitude mensuels

La plateforme couvre cinq secteurs métier :

| Secteur | Colonne cible | Colonnes clés |
|---------|--------------|---------------|
| 📱 Télécom | `Churn` | `tenure`, `MonthlyCharges`, `TotalCharges` |
| 💪 Salle de Sport | `resiliation` | `visites_mois`, `abonnement_mensuel`, `anciennete_mois` |
| 🛍️ E-commerce | `inactif` | `nb_commandes`, `panier_moyen`, `jours_inactif` |
| 🎓 EdTech | `desinscription` | `cours_termines`, `connexions_semaine`, `anciennete_mois` |
| ☁️ SaaS B2B | `resiliation` | `mrr`, `nb_utilisateurs`, `anciennete_mois` |

---

## 2. Nouveautés de la v2.0

| Fonctionnalité | Statut | Module(s) |
|---------------|--------|-----------|
| Programme de Fidélité complet (3 cohortes) | **NOUVEAU** | `loyalty_page.py`, `loyalty_config.py` |
| Catalogue de récompenses par secteur (5 secteurs × 2 types) | **NOUVEAU** | `loyalty_config.py` |
| Messages automatiques de gratitude mensuelle (Gmail SMTP) | **NOUVEAU** | `loyalty_messages_job.py` |
| Job APScheduler fidélité (1er du mois à 10h00) | **NOUVEAU** | `scheduler.py` |
| Panneau d'administration dynamique des récompenses | **NOUVEAU** | `loyalty_page.py` |
| Moteur de templates personnalisables (5 balises dynamiques) | **NOUVEAU** | `loyalty_page.py` |
| Mur des Champions avec cartes visuelles | **NOUVEAU** | `loyalty_page.py` |
| Anniversaires de contrat automatiques (`tenure % 12 == 0`) | **NOUVEAU** | `loyalty_messages_job.py` |
| Blank Slate (page d'accueil nouvel utilisateur) | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Simulateur What-If avec double jauge temps réel | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Page Alertes Clients avec export Excel ciblé (xlsxwriter) | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Assistant IA Chatbot contextuel | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Jauge de risque visuelle (Plotly Indicator) | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Moteur de Triage Statistique agnostique (P75) | **NOUVEAU** | `data_pipeline.py` |
| Migration auth SHA256 → bcrypt transparente | **AMÉLIORÉ** | `auth.py`, `database.py` |
| Navigation conditionnelle (Blank Slate vs modèle actif) | **AMÉLIORÉ** | `churn_prediction_dashboard.py` |
| Fallback local pour rapports PDF (si SendGrid absent) | **AMÉLIORÉ** | `email_reports.py` |
| Fuseau horaire Europe/Paris pour le scheduler | **AMÉLIORÉ** | `scheduler.py` |
| **Catalogue de récompenses dynamique (SQLite)** — création/suppression depuis l'UI | **NOUVEAU v2.1** | `loyalty_page.py`, `database.py` |
| **Webhook HTTP configurable** sur déclenchement de campagne (payload JSON structuré) | **NOUVEAU v2.1** | `loyalty_page.py` |
| **Simulateur What-If agnostique au secteur** — formulaire auto-généré depuis les features réelles | **AMÉLIORÉ v2.1** | `churn_prediction_dashboard.py` |
| **Détection cible améliorée** — `GLOBAL_TARGET_SYNONYMS` + nettoyage espaces + fallback interactif | **AMÉLIORÉ v2.1** | `data_pipeline.py` |
| **Visualisations agnostiques** — Overview, Visual Analytics, Alertes adaptés à tout secteur | **AMÉLIORÉ v2.1** | `churn_prediction_dashboard.py` |
| **Profil SHAP dynamique** — fallback sur 4 features numériques si champs standards absents | **AMÉLIORÉ v2.1** | `shap_explainer.py` |

---

## 3. Architecture générale

```
churn_prediction_dashboard.py  ← Point d'entrée Streamlit (routeur de pages)
│
├── auth.py                    ← Authentification (login / inscription / migration)
│   └── database.py            ← Couche SQLite (CRUD utilisateurs, sans dépendances)
│
├── data_pipeline.py           ← Import CSV · détection colonnes · nettoyage · XGBoost
│                                 · triage_risque() (moteur statistique agnostique P75)
│
├── shap_explainer.py          ← SHAP Tree Explainer · 4 vues · explication langage naturel
│
├── email_reports.py           ← PDF ReportLab · envoi SendGrid · fallback local
│
├── loyalty_page.py            ← Page Fidélité (3 cohortes, catalogue, config, mur champions)
│   └── loyalty_config.py      ← Catalogue récompenses · messages gratitude · seuils
│
├── scheduler.py               ← Singleton APScheduler (2 jobs cron)
│   ├── weekly_report_job.py   ← Job hebdo lundi 8h — PDF + SendGrid
│   └── loyalty_messages_job.py← Job mensuel 1er du mois 10h — Gmail SMTP
│
└── retainiq.db                ← Base SQLite (tables : users, reward_primitives)
```

**Flux de données principal :**

```
[Utilisateur] → auth.py → SQLite
     ↓
[CSV upload] → data_pipeline.py
     ├── detect_columns()     → classification automatique des colonnes
     ├── clean_data()         → imputation · encodage one-hot · normalisation cible
     ├── quality_report()     → score 0-100, avertissements, recommandations
     ├── train_custom_model() → XGBoost · sauvegarde model_[email].pkl + data_[email].csv
     └── triage_risque()      → enrichissement Motif de Risque + Action Suggérée (P75)
     ↓
[Dashboard] → prédictions ChurnProba · RiskLevel · Priorité · Motif de Risque
     ├── shap_explainer.py    → 4 vues SHAP
     ├── loyalty_page.py      → 3 cohortes + récompenses + simulation payload JSON
     └── email_reports.py     → PDF → SendGrid ou fallback local
     ↓
[APScheduler — 2 jobs]
     ├── Lundi 8h    → weekly_report_job.py → PDF → SendGrid
     └── 1er mois 10h → loyalty_messages_job.py → Gmail SMTP
```

---

## 4. Stack technique

| Couche | Technologie | Version recommandée | Rôle précis |
|--------|------------|---------------------|-------------|
| Framework web | Streamlit | 1.35+ | Rendu UI, session_state, routing pages |
| Machine Learning | XGBoost | 2.0+ | Classification binaire, `predict_proba()` |
| Prétraitement | scikit-learn | 1.4+ | `train_test_split`, métriques (F1, AUC-ROC) |
| Manipulation de données | Pandas | — | Lecture CSV, nettoyage, one-hot encoding |
| Calcul numérique | NumPy | — | Quantiles, matrices, opérations vectorielles |
| Explainabilité | SHAP | — | `TreeExplainer`, waterfall, valeurs Shapley |
| Visualisation interactive | Plotly | — | Gauges, histogrammes, scatter, pie, bar |
| Visualisation statique | Matplotlib / Seaborn | — | Corrélation, distribution (pages Analytics) |
| Base de données | SQLite (`sqlite3` stdlib) | — | Stockage utilisateurs, WAL mode |
| Auth / Hashing | bcrypt | — | Hash sécurisé des mots de passe + migration SHA256 |
| Génération PDF | ReportLab | — | Rapports A4 structurés (KPIs, Top 10 clients) |
| Export Excel avancé | xlsxwriter | — | Feuilles formatées avec en-têtes colorés, largeur auto |
| Email rapports | SendGrid API | — | Envoi PDF hebdomadaire avec pièce jointe |
| Email fidélité | Gmail SMTP (`smtplib`) | — | Messages de gratitude HTML/texte (SMTP_SSL 465) |
| Scheduling | APScheduler `BackgroundScheduler` | 3.x | 2 jobs cron (lundi 8h + 1er mois 10h) |
| Variables d'env | python-dotenv | — | Chargement `.env` (clés API, credentials SMTP) |

### Dépendances déclarées (`requirements.txt`)

```
streamlit · pandas · numpy · matplotlib · seaborn
scikit-learn · xgboost · plotly · bcrypt · shap
apscheduler · reportlab · python-dotenv · sendgrid · xlsxwriter
```

---

## 5. Structure des fichiers

```
AI-Powered-Churn-Prediction-main/
│
├── churn_prediction_dashboard.py   # ~1050 lignes — routeur Streamlit, toutes les pages
├── auth.py                         # Authentification, inscription, migration bcrypt
├── database.py                     # Couche SQLite (CRUD) — aucune dépendance projet
├── data_pipeline.py                # Pipeline ML : détection, nettoyage, entraînement, triage
├── shap_explainer.py               # Explainabilité SHAP (4 vues + langage naturel)
├── email_reports.py                # Génération PDF (ReportLab) + envoi SendGrid
├── weekly_report_job.py            # Job hebdomadaire : charge modèles → PDF → email
├── scheduler.py                    # Singleton APScheduler (2 jobs : hebdo + mensuel)
│
├── loyalty_page.py                 # Page Fidélité — 3 cohortes, carte champions, config
├── loyalty_config.py               # Catalogue récompenses, messages, seuils segmentation
├── loyalty_messages_job.py         # Job mensuel : filtrage champions → emails gratitude
│
├── retainiq.db                     # Base SQLite (auto-créée au premier démarrage)
├── requirements.txt                # Dépendances Python
├── .env                            # Variables d'environnement (non versionné)
├── .env.example                    # Template des variables d'environnement
├── CLAUDE.md                       # Instructions pour l'assistant IA de développement
├── SENDGRID_SETUP.md               # Guide de configuration SendGrid
│
├── model_[email_safe].pkl          # Modèle XGBoost par utilisateur (généré à l'usage)
├── data_[email_safe].csv           # Données nettoyées par utilisateur (généré à l'usage)
├── loyalty_settings.json           # Paramètres des campagnes par utilisateur (généré)
├── reports_archive/                # Dossier fallback PDF locaux (si SendGrid absent)
│
├── Telco-Customer-Churn.csv        # Dataset de démonstration (Kaggle Telco)
├── Churn_Prediction.ipynb          # Notebook d'exploration initiale
└── Cleaned_Data/                   # Données nettoyées pour le notebook
```

---

## 6. Schéma de la base de données

RetainIQ utilise SQLite via `retainiq.db`. La base est initialisée automatiquement à l'import du module `database.py`.

### Table `users`

| Colonne | Type | Contraintes | Description |
|---------|------|-------------|-------------|
| `email` | TEXT | PRIMARY KEY | Email professionnel (identifiant unique) |
| `password_hash` | TEXT | NOT NULL | Hash bcrypt (ou SHA256 pour anciens comptes) |
| `hash_type` | TEXT | NOT NULL, DEFAULT `'bcrypt'` | Type de hash : `'bcrypt'` ou `'sha256'` |
| `company` | TEXT | NOT NULL, DEFAULT `''` | Nom de l'entreprise |
| `secteur` | TEXT | NOT NULL, DEFAULT `''` | Secteur d'activité (ex: `'📱 Télécom'`) |
| `created_at` | TEXT | NOT NULL | Date de création ISO 8601 |

```sql
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    hash_type      TEXT NOT NULL DEFAULT 'bcrypt',
    company        TEXT NOT NULL DEFAULT '',
    secteur        TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL
);
```

### Table `reward_primitives`

| Colonne | Type | Contraintes | Description |
|---------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Identifiant unique auto-incrémenté |
| `user_email` | TEXT | NOT NULL | Email de l'utilisateur propriétaire |
| `label` | TEXT | NOT NULL | Nom personnalisé de la récompense |
| `action` | TEXT | NOT NULL DEFAULT `''` | Verbe d'action (ex : « Offrir », « Appliquer ») |
| `cible` | TEXT | NOT NULL DEFAULT `''` | Segment visé (ex : « Clients 12+ mois à risque ») |
| `valeur` | TEXT | NOT NULL DEFAULT `''` | Avantage consenti (ex : « -20% », « 1 mois gratuit ») |
| `duree` | TEXT | NOT NULL DEFAULT `''` | Durée de validité (ex : « 30 jours ») |
| `created_at` | TEXT | NOT NULL | Date de création ISO 8601 |

```sql
CREATE TABLE IF NOT EXISTS reward_primitives (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email  TEXT NOT NULL,
    label       TEXT NOT NULL,
    action      TEXT NOT NULL DEFAULT '',
    cible       TEXT NOT NULL DEFAULT '',
    valeur      TEXT NOT NULL DEFAULT '',
    duree       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
```

**Options de connexion :**
- `PRAGMA journal_mode=WAL` — mode Write-Ahead Logging pour la concurrence
- `row_factory = sqlite3.Row` — accès aux colonnes par nom

**Fichiers de données par utilisateur (filesystem) :**

| Fichier | Description |
|---------|-------------|
| `model_[email_safe].pkl` | Dictionnaire `{"model": XGBClassifier, "features": list}` |
| `data_[email_safe].csv` | DataFrame nettoyé (avec colonne `Churn` normalisée) |
| `loyalty_settings.json` | Paramètres de campagne par email d'utilisateur |

L'email est transformé en nom de fichier safe via : `email.replace("@", "_at_").replace(".", "_")`

---

## 7. Modèle de Données — Data Model

Cette section décrit formellement la structure des données à chaque étape du pipeline, depuis l'entrée brute jusqu'au jeu de données enrichi en sortie.

### 7.1 Structure d'entrée — CSV agnostique

Le fichier CSV importé par l'utilisateur doit respecter la structure minimale suivante. Les noms de colonnes sont détectés automatiquement par heuristiques ; seule la présence d'une variable cible binaire est strictement requise.

#### Colonnes obligatoires (par secteur)

| Secteur | Colonne cible acceptée | Colonnes numériques requises |
|---------|----------------------|------------------------------|
| 📱 Télécom | `Churn`, `churn`, `resiliation` | `tenure`, `MonthlyCharges`, `TotalCharges` |
| 💪 Salle de Sport | `resiliation`, `churn`, `depart` | `visites_mois`, `abonnement_mensuel`, `anciennete_mois` |
| 🛍️ E-commerce | `inactif`, `churn`, `depart` | `nb_commandes`, `panier_moyen`, `jours_inactif` |
| 🎓 EdTech | `desinscription`, `churn` | `cours_termines`, `connexions_semaine`, `anciennete_mois` |
| ☁️ SaaS B2B | `resiliation`, `churn`, `churned` | `mrr`, `nb_utilisateurs`, `anciennete_mois` |

#### Valeurs acceptées pour la variable cible

| Valeur brute | Encodage | Note |
|-------------|----------|------|
| `Yes`, `Oui`, `1`, `True`, `churned` | `1` | Client churné |
| `No`, `Non`, `0`, `False`, `active`, `actif` | `0` | Client actif |

#### Colonnes automatiquement ignorées

Toute colonne dont le nom contient l'un des mots-clés suivants est exclue du modèle :
`id`, `date`, `nom`, `name`, `email`, `phone`, `tel` — ainsi que toute colonne catégorielle présentant plus de 20 valeurs uniques distinctes.

#### Exemple de format minimal accepté

```csv
tenure,MonthlyCharges,TotalCharges,Contract,Churn
12,65.50,786.00,Month-to-month,No
3,89.00,267.00,Month-to-month,Yes
48,45.00,2160.00,Two year,No
7,92.00,644.00,Month-to-month,Yes
36,55.00,1980.00,One year,No
```

### 7.2 Transformations appliquées par le pipeline

| Étape | Transformation | Résultat |
|-------|---------------|---------|
| Colonnes numériques manquantes | Imputation par la médiane | Aucune valeur `NaN` résiduelle |
| Colonnes catégorielles manquantes | Imputation par le mode | Mode de la colonne |
| Encodage catégoriel | One-hot encoding (`pd.get_dummies`, `drop_first=True`) | Nouvelles colonnes binaires |
| Cible textuelle | Mapping Yes/No/Oui/Non → 1/0 | Colonne `Churn` binaire entière |
| Cible renommée | Renommage uniforme en `"Churn"` | Uniformité inter-secteurs |

### 7.3 Structure enrichie en sortie

Après exécution du pipeline de prédiction et du moteur de triage, le DataFrame est enrichi des colonnes suivantes :

| Colonne ajoutée | Type | Plage / Valeurs | Description |
|----------------|------|-----------------|-------------|
| `ChurnProba` | `float64` | `[0.0 ; 1.0]` | Probabilité de churn calculée par `model.predict_proba(X)[:, 1]` |
| `RiskLevel` | `str` | 3 niveaux | Libellé qualitatif dérivé de `ChurnProba` |
| `Priorité` | `str` | 3 niveaux | Label de priorisation opérationnelle pour les campagnes |
| `Motif de Risque` | `str` | 5 motifs | Cause principale identifiée par le moteur de triage statistique |
| `Action Suggérée` | `str` | Libre | Recommandation d'action associée au motif |

#### Détail des niveaux `RiskLevel`

| Seuil `ChurnProba` | Valeur `RiskLevel` | Couleur interface |
|-------------------|--------------------|-------------------|
| `> 0.60` | `"Risque élevé"` | Rouge `#EF4444` |
| `0.35 – 0.60` | `"Risque modéré"` | Orange `#F59E0B` |
| `≤ 0.35` | `"Risque faible"` | Vert `#00CC96` |

#### Détail des niveaux `Priorité`

| Seuil `ChurnProba` | Valeur `Priorité` |
|-------------------|--------------------|
| `> 0.80` | `"🔴 Critique (>80%)"` |
| `0.60 – 0.80` | `"🟠 Urgent (60-80%)"` |
| `0.40 – 0.60` | `"🟡 À suivre (40-60%)"` |

#### Catalogue des `Motif de Risque` (moteur de triage)

| Motif | Conditions de déclenchement |
|-------|-----------------------------|
| `"Nouveau client — pression tarifaire"` | `tenure ≤ 6` **ET** `MonthlyCharges > P75` |
| `"Absence d'engagement"` | Contrat mensuel détecté (one-hot ou colonne brute) |
| `"Pression tarifaire"` | `MonthlyCharges > P75` du dataset complet |
| `"Nouveau client"` | `tenure ≤ 6` uniquement |
| `"Risque d'insatisfaction globale"` | Aucune condition spécifique vérifiée |

---

## 8. Installation et démarrage

### Prérequis

- Python 3.11+
- pip

### Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd AI-Powered-Churn-Prediction-main

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

```bash
# Copier le template d'environnement
cp .env.example .env
# Éditer .env avec vos clés (voir section suivante)
```

### Lancement

```bash
streamlit run churn_prediction_dashboard.py
```

L'application s'ouvre sur `http://localhost:8501`.

### Commandes utilitaires

```bash
# Initialiser la base manuellement (déjà automatique)
python -c "from database import init_db; init_db()"

# Inspecter la base de données
sqlite3 retainiq.db ".schema"
sqlite3 retainiq.db "SELECT email, company, secteur FROM users;"
sqlite3 retainiq.db "SELECT user_email, label, action, valeur FROM reward_primitives;"

# Tester le job fidélité en standalone
python loyalty_messages_job.py

# Tester le job hebdomadaire en standalone
python weekly_report_job.py

# Migrer les anciens utilisateurs SHA256
python migrate_users.py
```

---

## 9. Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```dotenv
# ── SendGrid (rapports PDF hebdomadaires) ─────────────────────────
SENDGRID_API_KEY=SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SENDER_EMAIL=votre-expediteur@domaine.com
SENDER_NAME=RetainIQ

# ── Gmail SMTP (messages de fidélité mensuels) ────────────────────
GMAIL_ADDRESS=votre-adresse@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx    # Mot de passe d'application Gmail
```

| Variable | Usage | Obligatoire |
|----------|-------|-------------|
| `SENDGRID_API_KEY` | Envoi des rapports PDF hebdomadaires | Non (fallback local) |
| `SENDER_EMAIL` | Email expéditeur SendGrid | Oui si SendGrid activé |
| `SENDER_NAME` | Nom affiché dans les emails | Non (défaut : "RetainIQ") |
| `GMAIL_ADDRESS` | Compte Gmail pour les messages de fidélité | Non (simulation si absent) |
| `GMAIL_APP_PASSWORD` | Mot de passe d'application Gmail (2FA requis) | Non (simulation si absent) |

**Note :** Si `SENDGRID_API_KEY` est absent ou invalide, les rapports PDF sont automatiquement sauvegardés dans `reports_archive/` (fallback local). Si `GMAIL_ADDRESS` et `GMAIL_APP_PASSWORD` sont absents, les envois de fidélité sont simulés (succès fictif loggé).

---

## 10. Workflows Fonctionnels Détaillés

Cette section constitue la documentation de référence des quatre processus métier principaux de RetainIQ. Chaque workflow est décrit de manière exhaustive afin de servir de base directe à la rédaction du Cahier des Charges.

---

### 10.1 Workflow 1 — Ingestion et Prédiction ML

**Objectif :** Transformer un fichier CSV brut en prédictions de churn exploitables, avec un modèle XGBoost personnalisé par entreprise.

**Modules impliqués :** `data_pipeline.py`, `churn_prediction_dashboard.py`

#### Diagramme de séquence

```
[Utilisateur]        [Streamlit UI]         [data_pipeline.py]       [Filesystem]
      │                    │                        │                      │
      │── Upload CSV ──────►                        │                      │
      │                    │── pd.read_csv() ──────►│                      │
      │                    │                        │── detect_columns() ──►
      │                    │◄── detection_report ───│                      │
      │                    │── clean_data() ────────►│                      │
      │                    │◄── df_clean, log ───────│                      │
      │                    │── quality_report() ────►│                      │
      │                    │◄── score, issues ───────│                      │
      │◄── UI étapes 1-5 ──│                        │                      │
      │── Clic "Entraîner" ►                        │                      │
      │                    │── train_custom_model() ►│                      │
      │                    │                        │── model.fit() ────────►
      │                    │                        │── pickle.dump() ──────►model_[email].pkl
      │                    │                        │── df.to_csv() ────────►data_[email].csv
      │                    │◄── metrics (Acc, F1, AUC)                     │
      │◄── Résultats + 🎈 ─│                        │                      │
```

#### Étapes détaillées

**Étape 1 — Upload du fichier CSV**
- Widget `st.file_uploader(type=["csv"])` en interface Streamlit
- Lecture via `pd.read_csv(uploaded_file)`
- Affichage d'un aperçu brut (10 premières lignes) + métriques (lignes, colonnes, valeurs manquantes)

**Étape 2 — Détection automatique des colonnes** (`detect_columns(df, secteur)`)

La détection s'effectue selon l'algorithme suivant :

```
0. Nettoyer les espaces parasites en tête/queue de tous les noms de colonnes (str.strip())
1. Fusionner target_hints du secteur avec GLOBAL_TARGET_SYNONYMS (20+ synonymes universels)
   → Comparaison case-insensitive : "churn", "resiliation", "exited", "attrition", etc.
2. Si non trouvée → fallback interactif :
   a. Lister toutes les colonnes binaires (nunique() == 2)
   b. Si aucune → arrêt avec message d'erreur explicite
   c. Sinon → st.selectbox() pour que l'utilisateur désigne la colonne cible
      → renommer en "Churn" et relancer la détection
3. Classifier les colonnes restantes :
   a. Nom contient {id, date, nom, name, email, phone, tel} → ignorée
   b. dtype in [int64, float64]                             → numérique
   c. dtype == object AND nunique() <= 20                   → catégorielle
   d. dtype == object AND nunique() > 20                    → ignorée + warning
```

**Étape 3 — Nettoyage automatique** (`clean_data(df, detection_report)`)

| Opération | Méthode | Journalisé |
|-----------|---------|------------|
| Suppression colonnes ignorées | `df.drop(columns=ignored_cols)` | Oui |
| Imputation numériques | `df[col].fillna(df[col].median())` | Oui — valeur médiane |
| Imputation catégorielles | `df[col].fillna(df[col].mode()[0])` | Oui — valeur mode |
| Encodage one-hot | `pd.get_dummies(df, columns=cat_cols, drop_first=True)` | Oui |
| Normalisation cible | Mapping Yes/No → 1/0 via `str.lower().map(mapping)` | Oui |
| Renommage cible | `df.rename(columns={target_col: "Churn"})` | Oui |

**Étape 4 — Rapport de qualité** (`quality_report(df_raw, df_clean, detection_report)`)

Le score de qualité est calculé selon une grille de pénalités :

| Condition | Pénalité |
|-----------|---------|
| Taux de churn < 5% | −20 pts |
| Taux de churn > 60% | −15 pts |
| Valeurs manquantes > 20% | −20 pts |
| Valeurs manquantes > 5% | −5 pts |
| Dataset < 200 lignes | −30 pts |
| Dataset < 500 lignes | −10 pts |
| Colonne cible introuvable | −40 pts |

L'entraînement est **bloqué** si `score < 30`.

**Étape 5 — Visualisation pré-entraînement**
- 2 histogrammes Plotly (`barmode="overlay"`, `opacity=0.75`) des colonnes numériques les plus représentatives, colorés par `Churn` (vert = actif, rouge = churné)

**Étape 6 — Entraînement XGBoost** (`train_custom_model(df_clean, user_email)`)

```python
# Gestion automatique du déséquilibre de classes
scale_pos_weight = n_négatifs / n_positifs

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # calculé dynamiquement
)
```

Métriques retournées : **Accuracy**, **F1-Score** (macro), **AUC-ROC** — chacune en pourcentage arrondi à 2 décimales.

---

### 10.2 Workflow 2 — Moteur de Triage Statistique

**Objectif :** Enrichir les clients à haut risque d'un motif de risque explicatif et d'une action suggérée, sans recourir à aucune terminologie sectorielle codée en dur.

**Module impliqué :** `data_pipeline.py` — fonction `triage_risque(df_risque, df_full)`

#### Principe d'agnosticité sectorielle

Le moteur ne fait aucune hypothèse sur le secteur d'activité. Tous les seuils sont calculés **dynamiquement à partir des distributions réelles** du dataset courant :

- **Seuil tarifaire** : 75e percentile des charges mensuelles (`df_full['MonthlyCharges'].quantile(0.75)`)
- **Seuil d'ancienneté** : 6 mois (seuil universel du "nouveau client")
- **Détection d'engagement** : présence d'une colonne `Contract` (brute ou one-hot encodée)

```
charges_p75 = df_full['MonthlyCharges'].quantile(0.75)
```

Ce calcul est effectué sur le **dataset complet** (pas uniquement les clients à risque), afin de préserver la représentativité statistique de la distribution.

#### Matrice de décision du triage

```
                        Contrat mensuel ?
                       ┌──────┬───────┐
                       │  OUI │  NON  │
          ┌────────────┼──────┼───────┤
Nouveau   │  OUI       │  (1) │  (4)  │
client    ├────────────┼──────┼───────┤
(tenure≤6)│  NON       │  (2) │  (3)/(5)
          └────────────┴──────┴───────┘
                           ↑
                  Pression tarifaire (>P75) ?
```

| Cas | Motif attribué | Action suggérée |
|-----|---------------|-----------------|
| **(1)** nouveau + contrat mensuel + pression tarif | `"Nouveau client — pression tarifaire"` | Offre de bienvenue + options premium à prix réduit |
| **(2)** contrat mensuel uniquement | `"Absence d'engagement"` | Proposer un engagement long terme avec avantage tarifaire |
| **(3)** pression tarifaire uniquement | `"Pression tarifaire"` | Audit des services souscrits + offre d'optimisation de coût |
| **(4)** nouveau client uniquement | `"Nouveau client"` | Programme d'onboarding renforcé + contact personnalisé J+7 |
| **(5)** aucune condition | `"Risque d'insatisfaction globale"` | Enquête de satisfaction ciblée + appel de rétention sous 48h |

#### Détection du contrat mensuel (multi-format)

La colonne `Contract` peut être présente sous deux formes selon le niveau de preprocessing :

```python
# Format brut (avant one-hot encoding)
if 'Contract' in row.index:
    contrat_mensuel = 'month' in str(row['Contract']).lower()

# Format one-hot (après encoding)
else:
    one_yr = row.get('Contract_One_year', row.get('Contract_One year', None))
    two_yr = row.get('Contract_Two_year', row.get('Contract_Two year', None))
    if one_yr is not None and two_yr is not None:
        contrat_mensuel = (one_yr == 0) and (two_yr == 0)
```

---

### 10.3 Workflow 3 — Action en Boucle Fermée

**Objectif :** Transformer la prédiction ML en action opérationnelle concrète — ciblage croisé, sélection de récompense, simulation du déclenchement API et validation des garde-fous budgétaires.

**Modules impliqués :** `loyalty_page.py`, `loyalty_config.py`

#### Phase 1 — Enrichissement et filtrage des clients à risque

```
df_complet
    │
    ├── Filtre ChurnProba > 0.40 → clients_risque_raw
    │
    ├── triage_risque(clients_risque_raw, df_complet)
    │       → ajout colonnes : Motif de Risque · Action Suggérée
    │
    └── Calcul colonne Priorité :
            ChurnProba > 0.80 → 🔴 Critique (>80%)
            ChurnProba > 0.60 → 🟠 Urgent (60-80%)
            else              → 🟡 À suivre (40-60%)
```

#### Phase 2 — Ciblage croisé Priorité × Motif de Risque

L'interface expose deux filtres cumulatifs permettant un ciblage précis :

```
Filtre 1 : Priorité    ∈ {Tous | 🔴 Critique | 🟠 Urgent | 🟡 À suivre}
Filtre 2 : Motif       ∈ {Tous les motifs | [motifs détectés dynamiquement]}
                                                ↓
                              df_filtre = clients_enrichis
                                  .query(priorité)
                                  .query(motif)
```

Le résultat est affiché dans un tableau trié par score décroissant, limité aux 100 premiers clients, avec export CSV immédiat.

#### Phase 3 — Simulation du déclenchement API-First

Avant tout déclenchement, le système calcule dynamiquement le **Total Cumulé** de la campagne selon le type de récompense configuré :

```python
# Calcul du coût total estimé (extrait de loyalty_page.py)
if reward_type == "Pourcentage %":
    charges_moy   = df_filtre['MonthlyCharges'].clip(lower=0).mean()
    cout_unitaire = charges_moy * reward_value / 100.0
    total_cumule  = cout_unitaire * nb_filtres

elif reward_type == "Montant fixe €":
    cout_unitaire = reward_value
    total_cumule  = reward_value * nb_filtres

else:  # En nature
    total_cumule  = None   # Coût monétaire non quantifiable
```

Le **payload JSON** réellement envoyé au webhook configuré est structuré comme suit :

```json
{
  "event":     "loyalty_campaign_triggered",
  "timestamp": "2025-05-01T10:32:00",
  "company":   "Entreprise XYZ",
  "secteur":   "📱 Télécom",
  "reward": {
    "id":     3,
    "label":  "Cadeau Ancienneté",
    "action": "Offrir",
    "cible":  "Clients 12+ mois à risque",
    "valeur": "1 mois gratuit",
    "duree":  "30 jours"
  },
  "targeting": {
    "priority_filter": "🔴 Critique (>80%)",
    "motif_filter":    "Pression tarifaire",
    "total_clients":   12
  },
  "clients_sample": [
    {"churn_proba": 0.9142, "priorite": "🔴 Critique (>80%)", "motif": "Pression tarifaire"}
  ]
}
```

Le webhook est configuré depuis le panneau admin (Bloc 4 — Webhook). Si l'URL est vide ou invalide, le déclenchement de campagne reste fonctionnel mais sans appel réseau.

#### Phase 4 — Validation et déclenchement

```
Total Cumulé calculé
        │
        ├── total_cumule > budget_max ?
        │       OUI → Warning + bouton DÉSACTIVÉ
        │       NON → Bouton actif
        │
        ├── nb_filtres == 0 ?
        │       OUI → Info contextuelle + bouton DÉSACTIVÉ
        │       NON → continuer
        │
        └── Clic confirmé → st.success() + st.balloons()
                            (webhook HTTP à brancher en production)
```

---

### 10.4 Workflow 4 — Automatisation et Gouvernance

**Objectif :** Assurer l'exécution autonome et planifiée des tâches récurrentes (rapports, fidélité), la traçabilité des envois, et la gouvernance des exports de données.

**Modules impliqués :** `scheduler.py`, `weekly_report_job.py`, `loyalty_messages_job.py`, `churn_prediction_dashboard.py`

#### 4.1 Planification CRON (APScheduler)

Le scheduler est un **singleton thread-safe** initialisé une seule fois par processus Python. Il persiste entre tous les reruns Streamlit grâce au chargement unique des modules Python.

```python
# scheduler.py — pattern singleton
_scheduler = None
_lock = threading.Lock()

def start_scheduler(day_of_week="mon", hour=8, minute=0):
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler
        _scheduler = BackgroundScheduler(timezone="Europe/Paris")
        _scheduler.add_job(
            func=_run_weekly_job,
            trigger="cron",
            day_of_week=day_of_week,
            hour=hour, minute=minute,
            id="weekly_report",
            misfire_grace_time=3600,
            replace_existing=True,
        )
        _scheduler.add_job(
            func=send_loyalty_messages,
            trigger="cron",
            day=1, hour=10, minute=0,
            id="loyalty_messages",
            misfire_grace_time=3600,
            replace_existing=True,
        )
        _scheduler.start()
    return _scheduler
```

Les deux jobs enregistrés :

| ID Job | Déclencheur CRON | Fuseau | Tolérance misfire |
|--------|-----------------|--------|------------------|
| `weekly_report` | Lundi à 08:00 | Europe/Paris | 3 600 s (1 heure) |
| `loyalty_messages` | 1er du mois à 10:00 | Europe/Paris | 3 600 s (1 heure) |

**Initialisation dans le dashboard** (une seule fois par session Streamlit) :

```python
if not st.session_state.get("_scheduler_started"):
    sched.start_scheduler()
    st.session_state["_scheduler_started"] = True
```

#### 4.2 Gestion des destinataires via `session_state`

L'email de l'utilisateur connecté est persisté dans `st.session_state["user_email"]` dès la connexion. Il est utilisé comme valeur pré-remplie pour tous les envois manuels depuis le dashboard :

```python
# Page Alertes Clients — envoi manuel de rapport
recipient_email = st.text_input(
    "Email destinataire",
    value=st.session_state.get("user_email", ""),  # pré-rempli
    key="weekly_report_email"
)
```

Ce mécanisme garantit que l'utilisateur n'a jamais à ressaisir son email pour des actions répétées, tout en conservant la possibilité de modifier le destinataire à la volée.

#### 4.3 Export Excel avancé avec xlsxwriter

L'export des alertes clients utilise `xlsxwriter` (via `pd.ExcelWriter`) pour produire un fichier `.xlsx` formaté professionnellement :

```python
def _to_excel(dataframe: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df = dataframe.drop(columns=['ChurnProba', 'RiskLevel'], errors='ignore')
        export_df.to_excel(writer, index=False, sheet_name='Alertes')

        workbook  = writer.book
        worksheet = writer.sheets['Alertes']

        # En-têtes colorés (fond violet RetainIQ + police blanche + bordure)
        header_fmt = workbook.add_format({
            'bold':       True,
            'bg_color':   '#667eea',
            'font_color': 'white',
            'border':     1,
        })
        for col_num, col_name in enumerate(export_df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
            worksheet.set_column(col_num, col_num, max(18, len(str(col_name)) + 4))

    return output.getvalue()
```

Le nom de fichier inclut le seuil de risque et le motif filtré, permettant une traçabilité immédiate :

```
alertes_75pct_Pression_tarifaire.xlsx
```

#### 4.4 Rapport hebdomadaire automatique (job `weekly_report`)

```
send_weekly_reports()
    │
    ├── load_users()                  ← database.get_all_users()
    │
    └── Pour chaque utilisateur :
            │
            ├── load_user_model(email) ← model_[safe].pkl + data_[safe].csv
            │
            ├── model.predict_proba(X)[:, 1] → df['ChurnProba']
            │
            ├── generate_pdf_report(df, company, sector, path)
            │       → PDF A4 : KPIs + Top 10 clients + actions recommandées
            │
            └── send_pdf_via_sendgrid(to_email, subject, pdf_path)
                    │
                    ├── Succès → email envoyé
                    └── Échec  → _save_pdf_locally() → reports_archive/
```

#### 4.5 Job mensuel de gratitude (job `loyalty_messages`)

```
send_loyalty_messages()
    │
    ├── get_all_users() ← SQLite
    │
    └── Pour chaque utilisateur :
            │
            ├── _load_user_data(email) → (df, model, features)
            │
            ├── _filter_champions(df, model, features)
            │       ChurnProba < 0.20 ET tenure >= 12 → champions[]
            │
            └── Pour chaque champion :
                    │
                    ├── tenure % 12 == 0 → type_msg = "anniversaire" 🎂
                    ├── tenure % 12 != 0 → type_msg = "mensuel"      💌
                    │
                    ├── _build_message(i, tenure, secteur, type_msg)
                    │       → (subject, body) depuis GRATITUDE_MESSAGES
                    │         récompense auto : fidelite[i % len(fidelite)]
                    │
                    └── _send_gratitude_email(user_email, subject, body)
                            Gmail SMTP_SSL port 465
                            Simulation si GMAIL_ADDRESS absent
```

---

## 11. Workflow complet — étape par étape (v2.0)

### 11.1 Authentification et inscription

**Fichiers :** `auth.py`, `database.py`

1. L'utilisateur accède à l'application → `show_auth_page()` est appelée
2. **Connexion :** saisie email + mot de passe → `login_user(email, password)`
   - `get_user(email)` interroge SQLite
   - `_check_password()` vérifie selon le `hash_type` stocké
   - Si `hash_type == "sha256"` et mot de passe correct → migration transparente vers bcrypt via `update_user_hash()`
   - En cas de succès : injection dans `st.session_state` de `logged_in`, `user_email`, `user_company`, `user_secteur`
3. **Inscription :** saisie nom entreprise, email, secteur, mot de passe → `register_user()`
   - Vérification `user_exists(email)` → `create_user()` avec hash bcrypt
   - Validation : email unique, mots de passe identiques, minimum 6 caractères
4. **Déconnexion :** bouton sidebar → `logged_in = False`, rerun

```
[Page Login/Inscription]
        ↓
   login_user()
        ├── get_user()          ← SQLite SELECT
        ├── _check_password()   ← bcrypt.checkpw() ou SHA256
        └── update_user_hash()  ← migration transparente SHA256→bcrypt
        ↓
   st.session_state.logged_in = True
   st.rerun()
```

### 11.2 Blank Slate — premier démarrage

**Fichier :** `churn_prediction_dashboard.py`

Après connexion, le système vérifie l'existence du fichier `model_[email_safe].pkl` :

- **Sans modèle (Blank Slate) :** navigation réduite à `["🏠 Bienvenue", "📤 Importer mes données"]`. Un bandeau d'avertissement s'affiche dans la sidebar. La page Bienvenue guide l'utilisateur en 3 étapes visuelles.
- **Avec modèle actif :** navigation complète avec 11 pages. La sidebar affiche `✅ Modèle personnalisé actif`.

### 11.3 Import CSV et pipeline de données

**Fichier :** `data_pipeline.py` — Fonction principale : `show_pipeline_page(user_email, secteur)`

Le pipeline se déroule en **6 étapes** visuelles. Voir [Workflow 1](#101-workflow-1--ingestion-et-prédiction-ml) pour la description détaillée de chaque étape.

### 11.4 Entraînement du modèle XGBoost

**Fichier :** `data_pipeline.py` — `train_custom_model(df_clean, user_email)`

```python
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=n_neg / n_pos,  # auto-calculé
)
```

Le modèle entraîné est sérialisé avec pickle sous la forme :
```python
{"model": XGBClassifier, "features": list[str]}
```

### 11.5 Dashboard Overview et Visual Analytics

**Fichier :** `churn_prediction_dashboard.py`

- **Overview :** 4 KPIs (Total Clients, Taux Churn, Précision Modèle, Clients Urgents), tableau des 10 premiers clients avec score, export CSV complet. Les colonnes tenure et charges sont détectées dynamiquement par lookup dans une liste de synonymes (`tenure`, `anciennete_mois`, `mois_inscrit`, `mois_client` / `MonthlyCharges`, `abonnement_mensuel`, `mrr`, `panier_moyen`) — la page fonctionne quel que soit le secteur.
- **Visual Analytics :** Pie rétention/churn, histogramme des charges vs Churn *(conditionnel : affiché uniquement si une colonne de charges est détectée)*, boxplot ancienneté vs Churn *(conditionnel)*, top 10 features importance XGBoost, distribution churn (Seaborn), histogramme ancienneté *(conditionnel)*, matrice de corrélation. Chaque graphique conditionnel affiche un message `st.info` explicatif si la colonne correspondante est absente du dataset.

### 11.6 Prédiction IA et jauge de risque

**Fichier :** `churn_prediction_dashboard.py` — Page `🔮 AI Prediction`

Deux modes :
1. **Quick Prediction :** valeurs médianes automatiques pour toutes les features
2. **Manual Input :** layout fixe Telco (tenure, charges, contrat, internet) ou layout dynamique pour datasets custom (sliders générés automatiquement)

Résultat : jauge Plotly Indicator (mode gauge+number) colorée selon le niveau + bloc CSS `risk-high` / `risk-medium` / `risk-low` + recommandations contextuelles.

**Fonction `risk_gauge(score)` :**

| Score | Couleur | Label | Classe CSS |
|-------|---------|-------|-----------|
| > 0.60 | `#EF4444` (rouge) | RISQUE ÉLEVÉ | `risk-high` |
| 0.35–0.60 | `#F59E0B` (orange) | RISQUE MODÉRÉ | `risk-medium` |
| ≤ 0.35 | `#00CC96` (vert) | RISQUE FAIBLE | `risk-low` |

### 11.7 Simulateur What-If

**Fichier :** `churn_prediction_dashboard.py` — Page `⚡ Simulateur What-If`

Permet de comparer deux situations (avant/après une action commerciale). Le simulateur est **entièrement agnostique au secteur** : il génère automatiquement les contrôles (sliders, selectbox, number_input) à partir des features réelles du modèle entraîné.

1. **Analyse automatique des features** via `_whatsif_spec(col)` — classifie chaque colonne :
   - `constant` : valeur fixe, non modifiable
   - `binary` : selectbox Oui/Non (0/1)
   - `discrete` : selectbox de valeurs entières uniques (≤ 10 valeurs)
   - `continuous` : slider (min, max, médiane)
2. L'utilisateur configure la **situation actuelle** et la **situation cible** via des formulaires auto-générés côte à côte
3. `model.predict_proba(pd.DataFrame([row]))` calcule `score_a` et `score_b`
4. Affichage côte à côte : jauge avant, delta (+ ou −), jauge après
5. Économie client affichée si une colonne de charges est détectable parmi les inputs
6. Recommandations via `get_recommendations(score, tenure_val, charges_val)` avec lookup dynamique des colonnes tenure/charges

### 11.8 Alertes Clients

**Fichier :** `churn_prediction_dashboard.py` — Page `🚨 Alertes Clients`

- Slider de seuil (30%–90%) pour filtrer `df[df['ChurnProba'] > seuil]`
- Filtre additionnel par Motif de Risque (issu du moteur de triage)
- Tri par score décroissant, charges décroissantes ou ancienneté croissante — **avec fallback gracieux** : si la colonne correspondante est absente du dataset, le tri par score est appliqué automatiquement et un `st.info` en informe l'utilisateur
- KPIs : nombre de clients à risque, score moyen, revenu mensuel menacé *(affiché « N/A » si aucune colonne de charges n'est détectée)*
- Tableau paginé des 50 premiers + **export Excel formaté** (xlsxwriter, en-têtes colorés)
- Bloc envoi manuel de rapport PDF via SendGrid (avec pré-remplissage email `session_state`)
- 3 fiches d'actions recommandées (appel, offre, email)

### 11.9 Explainabilité SHAP

**Fichier :** `shap_explainer.py` — `show_shap_page(model, df, feature_names)`

**Vue 1 — Importance globale :** Barres horizontales des 15 features avec le plus grand `|SHAP moyen|`

**Vue 2 — Impact positif vs négatif :** Barres rouges (augmentent le churn) et vertes (le réduisent), avec ligne centrale à zéro

**Vue 3 — Explication individuelle :** Sélecteur de client (0–100), graphique waterfall SHAP, explication en langage naturel générée par `get_shap_explanation_text()`, profil complet du client. Les champs affichés (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) sont recherchés par nom exact ; si aucun n'est présent, un fallback dynamique sélectionne les 4 premières features numériques continues du modèle.

**Vue 4 — Scatter charges vs SHAP :** Nuage de points MonthlyCharges × Impact SHAP, coloré par score de risque

**Optimisation :** `compute_shap_values()` est décorée `@st.cache_data` pour éviter le recalcul à chaque interaction.

### 11.10 Assistant IA (Chatbot)

**Fichier :** `churn_prediction_dashboard.py` — `chatbot_response(question)`

Chatbot basé sur des règles avec correspondance de mots-clés en langue naturelle. L'historique est persisté dans `st.session_state.chat_history`.

| Mots-clés reconnus | Réponse |
|--------------------|---------|
| bonjour, salut, hello | Présentation + nombre de clients urgents |
| churn, partir, quitter | Définition + taux actuel + clients urgents |
| pourquoi, cause, raison | Top 3 facteurs de churn |
| xgboost, modèle, accuracy | Détails techniques du modèle |
| action, faire, retenir | Actions de rétention recommandées |
| simulateur, what-if | Présentation du simulateur |
| alerte, urgent | Nombre de clients en risque élevé |
| secteur, telecom, sport | Secteurs compatibles + secteur actif |
| combien, nombre, total | Statistiques de la base |
| (autre) | Message d'aide contextuel |

5 questions rapides pré-configurées en boutons.

### 11.11 Programme de Fidélité

**Fichiers :** `loyalty_page.py`, `loyalty_config.py`, `loyalty_messages_job.py`, `database.py`

La plateforme passe de l'IA **prédictive** à l'IA **prescriptive**. Deux améliorations majeures ont été apportées en v2.1 :

#### Catalogue de récompenses dynamique (SQLite)

Les récompenses ne sont plus une liste statique codée en dur. L'utilisateur crée et supprime ses propres récompenses depuis un formulaire dans la page Fidélité (`st.form("form_add_reward")`). Chaque récompense est stockée dans la table `reward_primitives` (voir Section 6) avec 4 primitives : **Action**, **Cible**, **Valeur**, **Durée**. Si le catalogue est vide, le bouton « Déclencher la campagne » est désactivé avec un message guidant l'utilisateur.

#### Webhook HTTP sur déclenchement de campagne

Un champ URL webhook est configurable dans le panneau admin (Bloc 4). À chaque déclenchement confirmé, `_send_webhook(url, payload)` effectue un `POST` JSON avec le payload structuré suivant :

```json
{
  "event":     "loyalty_campaign_triggered",
  "timestamp": "2025-05-01T10:32:00",
  "company":   "Entreprise XYZ",
  "secteur":   "📱 Télécom",
  "reward": {
    "id": 3, "label": "Cadeau Ancienneté",
    "action": "Offrir", "cible": "Clients 12+ mois à risque",
    "valeur": "1 mois gratuit", "duree": "30 jours"
  },
  "targeting": {
    "priority_filter": "🔴 Critique (>80%)",
    "motif_filter":    "Pression tarifaire",
    "total_clients":   12
  },
  "clients_sample": [
    {"churn_proba": 0.9142, "priorite": "🔴 Critique (>80%)", "motif": "Pression tarifaire"}
  ]
}
```

Le payload inclut un échantillon des 50 premiers clients ciblés. L'URL est sauvegardée dans `loyalty_settings.json` par utilisateur.

#### Segmentation en 3 cohortes (`segment_clients(df, secteur)`)

| Cohorte | Critères | Sous-labels | Action |
|---------|----------|-------------|--------|
| 🚨 **Cohorte A — Sauvetage** | `ChurnProba > 0.50` | 🔴 Critique / 🟠 Urgent / 🟡 À suivre | Récompenses de sauvetage immédiates |
| 🏆 **Cohorte B — Fidélité** | `ChurnProba < 0.35` ET `tenure ≥ Q75(tenure)` | 🥇 Légende / 🥈 Vétéran / 🥉 Fidèle | Récompenses de fidélité |
| 🌟 **Champions** | `ChurnProba < 0.20` ET `tenure ≥ 12 mois` | Mur des Champions | Messages automatiques mensuels |

**Balises dynamiques disponibles :** `{client_nom}`, `{anciennete_mois}`, `{valeur_recompense}`, `{score_risque}`, `{nom_entreprise}`

### 11.12 Rapports planifiés et scheduler

**Fichier :** `scheduler.py`

Le scheduler est un **singleton APScheduler** initialisé une seule fois par processus Python (thread-safe via `threading.Lock`).

**Status affiché :**
- Scheduler actif/inactif (vert/rouge)
- Prochaine exécution prévue
- Nombre de jobs enregistrés
- Historique des 10 dernières exécutions (date, statut, durée)

---

## 12. Documentation complète des modules

### `churn_prediction_dashboard.py`

Point d'entrée Streamlit (~1050 lignes). Gère la session, le CSS global, la navigation et le routage vers toutes les pages.

| Fonction / Section | Description |
|--------------------|-------------|
| `load_data()` | `@st.cache_data` — Charge `Telco-Customer-Churn.csv` ou génère 1000 clients synthétiques si absent |
| `train_model(df)` | `@st.cache_resource` — Entraîne XGBoost sur les données démo |
| `risk_gauge(score)` | Retourne `(fig, label, color, css_class)` — jauge Plotly Indicator tricolore |
| `get_recommendations(score, tenure, charges)` | Retourne 3 recommandations contextuelles selon le niveau de risque |
| `build_input_df(tenure, charges, contract, internet, security)` | Construit un DataFrame aligné sur `feature_names` pour le simulateur What-If |
| `chatbot_response(question)` | Moteur de réponses basé sur mots-clés — retourne une string markdown |
| `_to_excel(dataframe)` | Génère un fichier `.xlsx` formaté (xlsxwriter) — en-têtes colorés, largeur auto |
| Section `🏠 Bienvenue` | Page Blank Slate (sans modèle) — 3 cartes étapes + bouton navigation |
| Section `🏠 Overview` | KPIs + tableau 10 clients + export CSV + info dataset |
| Section `📊 Visual Analytics` | 7 graphiques Plotly/Matplotlib/Seaborn |
| Section `🔮 AI Prediction` | Prédiction manuelle/auto + jauge + recommandations |
| Section `🌟 Future Scenarios` | Simulation impact prix/tenure sur la distribution de risque |
| Section `⚡ Simulateur What-If` | Double jauge avant/après + delta + recommandations |
| Section `🚨 Alertes Clients` | Tableau filtrable + export Excel + envoi rapport PDF |
| Section `🤖 Assistant IA` | Chatbot avec historique session + questions rapides |
| Section `📤 Importer mes données` | Délégation à `show_pipeline_page()` |
| Section `🧠 Explainable AI` | Délégation à `show_shap_page()` |
| Section `⏰ Rapports Planifiés` | Config scheduler + envoi manuel + historique |
| Section `🏆 Programme de Fidélité` | Délégation à `show_loyalty_page()` |

**Classes CSS globales injectées :**

| Classe | Usage |
|--------|-------|
| `.main-header` | Bandeau dégradé violet en haut de page |
| `.metric-container` | Carte KPI rose/rouge dégradée |
| `.insight-card` | Carte bleu/cyan |
| `.prediction-card` | Carte prédiction |
| `.risk-high` | Fond rouge sombre + bordure rouge |
| `.risk-medium` | Fond orange sombre + bordure orange |
| `.risk-low` | Fond vert sombre + bordure verte |
| `.section-card` | Carte générique fond sombre |
| `.alert-box` | Encadré alerte fond rouge avec bordure gauche |
| `.chat-user` | Bulle utilisateur alignée à droite |
| `.chat-bot` | Bulle bot alignée à gauche |
| `.secteur-badge` | Badge secteur sidebar |

---

### `auth.py`

Gestion complète de l'authentification. Dépend de `database.py`. Ne persiste aucun état lui-même.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `hash_password` | `(password: str) → str` | Hash bcrypt avec sel aléatoire. Retourne chaîne UTF-8. |
| `_check_password` | `(password, stored_hash, hash_type) → bool` | Vérifie selon `hash_type` : `'bcrypt'` → `bcrypt.checkpw()`, `'sha256'` → SHA256 hexdigest |
| `load_users` | `() → dict` | Compatibilité `weekly_report_job.py` — délègue à `get_all_users()` |
| `register_user` | `(email, password, company, secteur) → (bool, str)` | Crée un compte bcrypt. Retourne `(True, msg_succès)` ou `(False, msg_erreur)` |
| `login_user` | `(email, password) → (bool, dict|str)` | Vérifie identifiants. Migration SHA256→bcrypt si besoin. Retourne `(True, {company, secteur, created_at})` ou `(False, erreur)` |
| `show_auth_page` | `()` | Affiche la page Streamlit Login/Inscription avec 2 onglets et formulaires |

**Migration transparente SHA256 → bcrypt :** Lors d'une connexion réussie avec un ancien compte SHA256, `update_user_hash()` est appelé automatiquement pour stocker le nouveau hash bcrypt. L'utilisateur ne voit rien.

---

### `database.py`

Couche d'accès SQLite pure — aucune dépendance vers les autres modules du projet. Initialisée automatiquement à l'import via `init_db()`.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `get_connection` | `() → contextmanager` | Context manager SQLite avec WAL mode, row_factory, commit/rollback automatique |
| `init_db` | `()` | `CREATE TABLE IF NOT EXISTS users` + `reward_primitives` — idempotent, appelé à l'import |
| `get_user` | `(email: str) → dict | None` | `SELECT * FROM users WHERE email = ?` — retourne dict ou None |
| `create_user` | `(email, password_hash, company, secteur, hash_type) → None` | `INSERT INTO users` — lève `ValueError` si email déjà utilisé |
| `update_user_hash` | `(email, new_hash, new_type) → None` | `UPDATE users SET password_hash, hash_type` — migration bcrypt |
| `get_all_users` | `() → dict` | Retourne tous les users sous forme `{email: {company, secteur, created_at}}` |
| `user_exists` | `(email: str) → bool` | Alias de `get_user(email) is not None` |
| `get_reward_primitives` | `(user_email: str) → list[dict]` | Retourne toutes les récompenses de l'utilisateur triées par `id` |
| `create_reward_primitive` | `(user_email, label, action, cible, valeur, duree) → int` | Insère une récompense, retourne l'`id` auto-incrémenté |
| `delete_reward_primitive` | `(primitive_id: int) → None` | Supprime la récompense identifiée par son `id` |

---

### `data_pipeline.py`

Pipeline ML complet : de l'upload CSV à l'entraînement XGBoost. Contient aussi `show_pipeline_page()` qui orchestre les 6 étapes Streamlit, et `triage_risque()` le moteur de triage statistique agnostique.

#### Constante `SECTEUR_COLUMNS`

Dictionnaire de configuration par secteur définissant `required` (colonnes requises), `target_hints` (noms possibles de la colonne cible), `description` (texte d'aide).

#### Constante `GLOBAL_TARGET_SYNONYMS`

Liste de 20+ synonymes universels pour la détection de la colonne cible, fusionnée avec les `target_hints` du secteur : `"churn"`, `"resiliation"`, `"résiliation"`, `"exited"`, `"status"`, `"attrition"`, `"churned"`, `"inactif"`, `"desinscription"`, etc.

#### Fonctions

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `detect_columns` | `(df, secteur) → dict` | Détecte `target_col`, `numeric_cols`, `categorical_cols`, `ignored_cols`, `warnings`. Nettoie d'abord les espaces parasites des noms de colonnes. Fusionne `target_hints` secteur + `GLOBAL_TARGET_SYNONYMS`. Si aucune cible trouvée, renvoie un rapport avec `target_col=None` pour déclencher le fallback interactif. |
| `clean_data` | `(df, detection_report) → (df_clean, cleaning_log)` | Supprime colonnes ignorées, impute numériques (médiane), encode catégorielles (one-hot), mappe cible (Yes/No → 1/0), renomme en `"Churn"`. |
| `quality_report` | `(df_raw, df_clean, detection_report) → dict` | Score 0-100 avec pénalités : churn < 5% (−20), churn > 60% (−15), missing > 20% (−20), missing > 5% (−5), < 200 lignes (−30), < 500 lignes (−10), pas de cible (−40). |
| `train_custom_model` | `(df_clean, user_email) → (model, metrics, error)` | Split stratifié 80/20, XGBoost avec `scale_pos_weight` auto, sauvegarde `.pkl` et `.csv`. Retourne `(None, None, msg_erreur)` si problème. |
| `load_user_model` | `(user_email) → (model, features, df)` | Charge `model_[safe].pkl` et `data_[safe].csv`. Retourne `(None, None, None)` si absent. |
| `triage_risque` | `(df_risque, df_full) → df` | Moteur de triage statistique (P75). Ajoute `Motif de Risque` et `Action Suggérée`. Agnostique au secteur. |
| `show_pipeline_page` | `(user_email, secteur)` | Page Streamlit complète — orchestre les 6 étapes avec UI progressive. Gère le fallback interactif de sélection de colonne cible : si `detect_columns()` ne trouve pas de cible, liste les colonnes binaires disponibles et propose un `st.selectbox`. |

---

### `shap_explainer.py`

Module d'explainabilité SHAP pour XGBoost.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `compute_shap_values` | `(_model, _X) → (shap_values, expected_value)` | `@st.cache_data` — `shap.TreeExplainer` + `.shap_values()`. Préfixe `_` pour exclure du cache key. |
| `get_shap_explanation_text` | `(shap_vals, feature_names, top_n=3) → str` | Génère explication langage naturel via `LABEL_MAP`. Identifie top facteurs positifs et protecteurs. |
| `show_shap_page` | `(model, df, feature_names)` | Page complète 4 vues : importance globale, impact signé, waterfall individuel, scatter charges/SHAP. |

**`LABEL_MAP` :** Dictionnaire de 20+ entrées mappant noms techniques vers labels lisibles (ex: `"Contract_Month-to-month"` → `"le contrat mensuel"`).

**4 vues détaillées :**

| Vue | Type graphique | Description |
|-----|----------------|-------------|
| 1 | Bar horizontal | `mean(|SHAP|)` — 15 features les plus importantes |
| 2 | Bar horizontal bicolore | `mean(SHAP)` signé — rouge si augmente churn, vert si réduit |
| 3 | Waterfall individuel | `shap_values[client_idx]` — 12 features triées par impact absolu |
| 4 | Scatter | `MonthlyCharges` × `SHAP(MonthlyCharges)` coloré par risque |

---

### `email_reports.py`

Génération PDF (ReportLab) et envoi email (SendGrid). Fallback local si SendGrid non configuré.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `_safe` | `(v) → str` | Formate une valeur pour PDF (NaN → "-", float → 2 décimales) |
| `_risk_label` | `(score: float) → str` | Texte du niveau de risque : "Risque élevé" / "Risque modéré" / "Risque faible" |
| `prepare_scored_df` | `(df) → df` | Vérifie `ChurnProba` présente, ajoute `RiskLevel` si absent |
| `generate_pdf_report` | `(df, company_name, sector, output_path, report_title) → str` | Génère PDF A4 : titre, KPIs (tableau), Top 10 clients à risque, actions recommandées. Retourne `output_path`. |
| `_save_pdf_locally` | `(pdf_path, to_email) → str` | Fallback : copie dans `reports_archive/report_[safe_email]_[timestamp].pdf` |
| `send_pdf_via_sendgrid` | `(to_email, subject, body_text, pdf_path, from_email, from_name) → (bool, str)` | Envoie via API SendGrid avec PDF en pièce jointe. 3 cas de fallback → sauvegarde locale. |

**Structure du PDF généré :**
1. En-tête (titre, entreprise, secteur, date de génération)
2. Tableau KPIs (total clients, taux churn, risque élevé/modéré/faible)
3. Tableau Top 10 clients (client, ancienneté, charges, total, score %, niveau)
4. Actions recommandées (4 points)
5. Pied de page "Généré automatiquement par RetainIQ"

---

### `weekly_report_job.py`

Job exécuté automatiquement chaque lundi à 8h00 (fuseau Europe/Paris).

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `send_weekly_reports` | `()` | Pour chaque utilisateur : charge modèle → calcule `ChurnProba` → génère PDF → envoie via SendGrid. Skip si pas de modèle/données. |

**Flux d'exécution :**

```
load_users()           ← auth.py → database.py
    ↓
for each user:
    load_user_model(email)   ← data_pipeline.py
    predict_proba(X)         ← calcul ChurnProba si absent
    generate_pdf_report()    ← email_reports.py
    send_pdf_via_sendgrid()  ← email_reports.py (fallback local si erreur)
```

**Variables d'environnement requises :** `SENDER_EMAIL`, `SENDGRID_API_KEY`

---

### `scheduler.py`

Singleton APScheduler thread-safe. Persiste entre les reruns Streamlit car Python ne recharge pas les modules.

| Variable globale | Type | Description |
|-----------------|------|-------------|
| `_scheduler` | `BackgroundScheduler | None` | Instance unique du scheduler |
| `_lock` | `threading.Lock` | Protection thread-safe pour l'initialisation |
| `run_history` | `list[dict]` | Historique des exécutions (max 50, ordre chronologique inverse) |
| `next_run_time` | `datetime | None` | Prochaine exécution planifiée du job hebdomadaire |

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `_run_weekly_job` | `()` | Wrapper APScheduler : appelle `send_weekly_reports()`, logue durée/statut dans `run_history` |
| `start_scheduler` | `(day_of_week, hour, minute) → BackgroundScheduler` | Démarre le scheduler si non actif. Enregistre 2 jobs : `weekly_report` et `loyalty_messages`. Fuseau : Europe/Paris. |
| `stop_scheduler` | `()` | Arrête proprement via `shutdown(wait=False)` |
| `get_status` | `() → dict` | Retourne `{running, next_run, job_count, history[:10]}` |
| `trigger_now` | `()` | Lance `_run_weekly_job()` immédiatement (test manuel depuis le dashboard) |
| `update_schedule` | `(day_of_week, hour, minute)` | `reschedule_job("weekly_report", ...)` sans redémarrage. Démarre si inactif. |
| `_refresh_next_run` | `()` | Met à jour `next_run_time` depuis `_scheduler.get_job("weekly_report")` |

---

### `loyalty_page.py`

Page Streamlit complète du Programme de Fidélité. Gère l'enrichissement triage, les filtres de ciblage croisé, le catalogue de récompenses dynamique (SQLite), le déclenchement webhook et la configuration des campagnes.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `_load_settings` | `(user_email) → dict` | Charge depuis `loyalty_settings.json` les paramètres de l'utilisateur. Merge avec `_DEFAULT_SETTINGS`. |
| `_save_settings` | `(user_email, settings) → None` | Sauvegarde dans `loyalty_settings.json` au format `{email: settings}`. |
| `_send_webhook` | `(url: str, payload: dict) → (bool, str)` | POST JSON vers `url` (timeout 10 s). Retourne `(True, "HTTP 200")` ou `(False, message_erreur)`. Rejette les URLs non-HTTP. |
| `segment_clients` | `(df, secteur) → (cohorte_a, cohorte_b, champions)` | Retourne 3 DataFrames selon `SEGMENTATION_CONFIG`. Ajoute `Priorité` (A) et `Médaille` (B). |
| `show_loyalty_page` | `(df, secteur, user_company, user_email)` | Page principale : KPIs, tableau de ciblage croisé, catalogue de récompenses dynamique, déclenchement campagne avec webhook, config. |
| `_render_config_panel` | `(user_email, user_company, secteur)` | Panneau expander de 4 blocs (seuils, valeur, garde-fous, **webhook**) avec formulaire Streamlit. |

**Paramètres par défaut `_DEFAULT_SETTINGS` :**

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `seuil_urgence` | `0.65` | Seuil Cohorte A (65%) |
| `tenure_min_b` | `12` | Ancienneté min Cohorte B |
| `depense_min` | `0.0` | Dépense minimum |
| `reward_type` | `"Pourcentage %"` | Type de récompense |
| `reward_value` | `20.0` | Valeur de la récompense |
| `email_subject` | Template avec balise `{client_nom}` | Objet du message |
| `email_body` | Template complet | Corps du message |
| `budget_max` | `5000.0` | Budget mensuel max |
| `quota_mois` | `100` | Quota mensuel |
| `periode_carence` | `3` | Mois entre deux récompenses |
| `campagne_sauvetage` | `True` | Activation campagne A |
| `campagne_fidelite` | `True` | Activation campagne B |
| `webhook_url` | `""` | URL webhook de déclenchement campagne (vide = désactivé) |

---

### `loyalty_config.py`

Fichier de configuration pur — pas de code Streamlit, pas de dépendances externes.

#### `REWARDS_CATALOG`

Dictionnaire imbriqué `secteur → {sauvetage: [...], fidelite: [...], seuil_sauvetage, seuil_fidelite, tenure_fidelite, devise}`.

| Secteur | Récompenses sauvetage (extrait) | Récompenses fidélité (extrait) |
|---------|--------------------------------|-------------------------------|
| 📱 Télécom | Pass Internet 5Go, -30% sur facture, gel 1 mois | Double/Triple points Club, surclassement, VIP |
| 💪 Sport | Gel 1-2 mois, coaching gratuit, nutrition 1 mois | Pass invité, accès spa, cours illimités |
| 🛍️ E-commerce | -15/-20% 48h, livraison express, cadeau surprise | Accès soldes anticipé, cashback 5%, badge VIP |
| 🎓 EdTech | Gel compte 1 mois, -25% renouvellement, coaching | Certificat excellence, accès early bird, badge |
| ☁️ SaaS B2B | Gel tarif 12 mois, audit offert, Premium 2 mois | Étude de cas, 2 licences gratuites, co-marketing |

#### `GRATITUDE_MESSAGES`

Templates par secteur × type (`anniversaire` / `mensuel`). Variables dynamiques : `{annees}`, `{tenure}`, `{reward}`.

#### `SEGMENTATION_CONFIG`

```python
{
    "champion_proba_max":   0.20,   # ChurnProba < 20% pour Champion
    "champion_tenure_min":  12,     # Tenure >= 12 mois pour Champion
    "sauvetage_proba_min":  0.50,   # ChurnProba > 50% pour Cohorte A
    "fidelite_proba_max":   0.35,   # ChurnProba < 35% pour Cohorte B
}
```

---

### `loyalty_messages_job.py`

Job mensuel (1er du mois à 10h00) d'envoi des messages de gratitude aux Champions.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `_load_users` | `() → dict` | Délègue à `database.get_all_users()` |
| `_load_user_data` | `(user_email) → (df, model, features)` | Charge `data_[safe].csv` et `model_[safe].pkl`. Retourne `(None, None, None)` si absent. |
| `_filter_champions` | `(df, model, features) → df` | Calcule `ChurnProba` si absent, filtre `ChurnProba < 0.20` ET `tenure >= 12`. |
| `_build_message` | `(client_idx, tenure, secteur, type_msg) → (subject, body)` | Choisit template `anniversaire` ou `mensuel` depuis `GRATITUDE_MESSAGES`. Sélectionne récompense par `client_idx % len(rewards)`. |
| `_send_gratitude_email` | `(to_email, subject, body_text) → bool` | Gmail SMTP via `smtplib.SMTP_SSL('smtp.gmail.com', 465)`. Corps texte + HTML. Retourne `True` si succès ou si Gmail non configuré (simulation). |
| `send_loyalty_messages` | `() → dict` | Job principal : itère tous les utilisateurs → champions → messages. Retourne `{status, total_sent, anniversaires, mensuels, erreurs, executed_at}`. |

**Logique anniversaire :**
```python
if tenure % 12 == 0 and tenure > 0:
    type_msg = "anniversaire"   # 12 mois, 24 mois, 36 mois...
else:
    type_msg = "mensuel"        # Reconnaissance mensuelle classique
```

**Format email HTML généré :** Fond sombre `#0A1628`, logo RetainIQ en vert `#02C39A`, corps avec bordure verte, pied de page daté automatiquement.

---

## 13. Dictionnaire des Fonctions Core

Cette section recense les fonctions principales de l'application, leur rôle métier exact et leurs contrats d'interface, afin de servir de référence pour la maintenance et les évolutions.

### 13.1 Pipeline de données et Machine Learning

| Fonction | Module | Rôle métier | Entrée | Sortie |
|----------|--------|-------------|--------|--------|
| `detect_columns` | `data_pipeline` | Inférer automatiquement la structure du dataset client | `(df: DataFrame, secteur: str)` | `dict` avec clés `target_col`, `numeric_cols`, `categorical_cols`, `ignored_cols`, `warnings` |
| `clean_data` | `data_pipeline` | Normaliser le dataset pour l'entraînement ML | `(df, detection_report: dict)` | `(df_clean: DataFrame, cleaning_log: list[str])` |
| `quality_report` | `data_pipeline` | Évaluer la fiabilité du dataset avant entraînement | `(df_raw, df_clean, detection_report)` | `dict` avec `score`, `issues`, `recommendations`, `churn_rate` |
| `train_custom_model` | `data_pipeline` | Entraîner un modèle XGBoost personnalisé par entreprise | `(df_clean: DataFrame, user_email: str)` | `(model, metrics: dict, error: str\|None)` |
| `load_user_model` | `data_pipeline` | Charger le modèle actif de l'utilisateur depuis le filesystem | `(user_email: str)` | `(model, features: list, df: DataFrame)` ou `(None, None, None)` |
| `triage_risque` | `data_pipeline` | Attribuer un motif de risque et une action aux clients à haut risque | `(df_risque: DataFrame, df_full: DataFrame)` | `df` enrichi avec `Motif de Risque` et `Action Suggérée` |

### 13.2 Authentification et gestion des utilisateurs

| Fonction | Module | Rôle métier | Entrée | Sortie |
|----------|--------|-------------|--------|--------|
| `login_user` | `auth` | Authentifier un utilisateur et migrer son hash si SHA256 | `(email: str, password: str)` | `(True, {company, secteur, created_at})` ou `(False, erreur: str)` |
| `register_user` | `auth` | Créer un nouveau compte avec hash bcrypt | `(email, password, company, secteur)` | `(bool, message: str)` |
| `hash_password` | `auth` | Hacher un mot de passe en bcrypt | `(password: str)` | `hashed: str` |
| `get_user` | `database` | Récupérer un utilisateur par email | `(email: str)` | `dict` ou `None` |
| `get_all_users` | `database` | Lister tous les utilisateurs (pour les jobs planifiés) | `()` | `{email: {company, secteur, created_at}}` |

### 13.3 Génération de rapports

| Fonction | Module | Rôle métier | Entrée | Sortie |
|----------|--------|-------------|--------|--------|
| `generate_pdf_report` | `email_reports` | Produire un rapport PDF A4 structuré avec KPIs et Top 10 | `(df, company_name, sector, output_path, report_title)` | `output_path: str` |
| `send_pdf_via_sendgrid` | `email_reports` | Envoyer le rapport PDF par email via l'API SendGrid | `(to_email, subject, body_text, pdf_path, from_email, from_name)` | `(bool, message: str)` |
| `_to_excel` | `churn_prediction_dashboard` | Exporter les alertes clients en Excel formaté (xlsxwriter) | `(dataframe: DataFrame)` | `bytes` (contenu `.xlsx`) |

### 13.4 Explainabilité

| Fonction | Module | Rôle métier | Entrée | Sortie |
|----------|--------|-------------|--------|--------|
| `compute_shap_values` | `shap_explainer` | Calculer les valeurs SHAP via TreeExplainer (mise en cache) | `(_model, _X: DataFrame)` | `(shap_values: ndarray, expected_value: float)` |
| `get_shap_explanation_text` | `shap_explainer` | Générer une explication en langage naturel des facteurs de risque | `(shap_vals, feature_names, top_n=3)` | `explication: str` |

### 13.5 Fidélisation et automatisation

| Fonction | Module | Rôle métier | Entrée | Sortie |
|----------|--------|-------------|--------|--------|
| `segment_clients` | `loyalty_page` | Classer tous les clients en 3 cohortes de fidélité | `(df, secteur: str)` | `(cohorte_a, cohorte_b, champions)` — 3 DataFrames |
| `_load_settings` | `loyalty_page` | Charger les règles de campagne de l'utilisateur | `(user_email: str)` | `dict` (fusionné avec `_DEFAULT_SETTINGS`) |
| `_send_webhook` | `loyalty_page` | Envoyer le payload de campagne en POST JSON vers une URL configurée | `(url: str, payload: dict)` | `(bool, message: str)` |
| `get_reward_primitives` | `database` | Lister les récompenses de l'utilisateur depuis SQLite | `(user_email: str)` | `list[dict]` |
| `create_reward_primitive` | `database` | Créer une récompense dans le catalogue SQLite | `(user_email, label, action, cible, valeur, duree)` | `int` (id créé) |
| `delete_reward_primitive` | `database` | Supprimer une récompense du catalogue | `(primitive_id: int)` | `None` |
| `send_loyalty_messages` | `loyalty_messages_job` | Envoyer les messages de gratitude mensuels aux Champions | `()` | `{status, total_sent, anniversaires, mensuels, erreurs, executed_at}` |
| `send_weekly_reports` | `weekly_report_job` | Générer et envoyer les rapports PDF hebdomadaires | `()` | `None` (effets de bord : emails + PDF locaux) |
| `start_scheduler` | `scheduler` | Initialiser le singleton APScheduler avec 2 jobs CRON | `(day_of_week, hour, minute)` | `BackgroundScheduler` |
| `trigger_now` | `scheduler` | Déclencher manuellement le job hebdomadaire pour test | `()` | `None` |

---

## 14. Pages du dashboard (toutes)

| Page | Condition d'accès | Description |
|------|-------------------|-------------|
| `🏠 Bienvenue` | Sans modèle (Blank Slate) | Guide d'onboarding en 3 étapes, bouton vers import |
| `📤 Importer mes données` | Toujours disponible | Pipeline CSV complet en 6 étapes |
| `🏠 Overview` | Avec modèle | KPIs + tableau 10 clients + export |
| `📊 Visual Analytics` | Avec modèle | 7 graphiques analytiques |
| `🔮 AI Prediction` | Avec modèle | Prédiction manuelle ou auto avec jauge |
| `🌟 Future Scenarios` | Avec modèle | Simulation impact prix/tenure |
| `⚡ Simulateur What-If` | Avec modèle | Comparaison avant/après action |
| `🚨 Alertes Clients` | Avec modèle | Filtrage clients à risque + export Excel |
| `🤖 Assistant IA` | Avec modèle | Chatbot contextuel |
| `🧠 Explainable AI` | Avec modèle | 4 vues SHAP |
| `⏰ Rapports Planifiés` | Avec modèle | Config scheduler + historique |
| `🏆 Programme de Fidélité` | Avec modèle | 3 cohortes + récompenses + config |

---

## 15. Système de segmentation clients v2.0

Le système classe automatiquement tous les clients en 4 groupes à partir de deux axes : **score de risque churn** (ChurnProba) et **ancienneté** (tenure).

```
ChurnProba
    │
100%├─────────────────────────────────────────────────────────
    │              🚨 Cohorte A — SAUVETAGE
 50%├───────────────────────────────────────────────────────── ← seuil_sauvetage
    │                    📊 Autres
 35%├───────────────────────────────────────────────────────── ← seuil_fidelite
    │   📊 Autres              │        🏆 Cohorte B
 20%├──────────────────────── ─ ─ ─ ─ ──────────────────────── ← champion_proba_max
    │              🌟 Champions
  0%└─────────────────────────────────────────────────────────── tenure
         0m        12m        Q75         60m
                   ↑                      ↑
            champion_tenure_min      Légende (Cohorte B)
```

**Règles de priorité Cohorte A :**

| Score | Priorité |
|-------|----------|
| ChurnProba > 80% | 🔴 Critique |
| 65% < ChurnProba <= 80% | 🟠 Urgent |
| 50% < ChurnProba <= 65% | 🟡 À suivre |

**Médailles Cohorte B :**

| Ancienneté | Médaille |
|-----------|----------|
| >= 60 mois | 🥇 Légende |
| 36–59 mois | 🥈 Vétéran |
| < 36 mois | 🥉 Fidèle |

---

## 16. Catalogue de récompenses

### 16.1 Catalogue statique par secteur (`loyalty_config.py`)

Définit **5 secteurs × 2 types × 6 à 8 récompenses** chacun, soit 60+ récompenses préconfigurées adaptées au **marché marocain** (devise MAD). Ce catalogue sert de référence et de source pour le job de fidélité mensuel (`loyalty_messages_job.py`).

### 16.2 Catalogue dynamique par utilisateur (SQLite)

Chaque utilisateur peut créer son propre catalogue de récompenses personnalisées via l'UI (panneau « Gérer le Catalogue » dans la page Fidélité). Ces récompenses sont stockées dans la table `reward_primitives` et sont les seules proposées lors du déclenchement manuel d'une campagne. Structure d'une récompense : **Label** (identifiant unique) + 4 primitives (**Action**, **Cible**, **Valeur**, **Durée**).

> **Note :** Les deux catalogues coexistent. Le catalogue statique alimente les jobs automatiques (Messages de Champions) ; le catalogue dynamique alimente les déclenchements manuels depuis le dashboard.

Chaque secteur définit également :
- `seuil_sauvetage` : `0.50` (universel)
- `seuil_fidelite` : `0.35` (universel)
- `tenure_fidelite` : ancienneté minimale pour la Cohorte B (6 mois E-commerce/EdTech, 12 mois Sport/SaaS B2B, 18 mois Télécom)
- `devise` : `MAD`

Les messages de gratitude (`GRATITUDE_MESSAGES`) sont personnalisés par secteur — ton "membre" pour le sport, "apprenant" pour EdTech, "partenaire" pour SaaS B2B, "client fidèle" pour E-commerce et Télécom.

---

## 17. Scheduler — deux jobs automatiques

| Job | ID | Déclencheur | Fonction | Description |
|-----|----|-------------|----------|-------------|
| Rapport hebdomadaire | `weekly_report` | Lundi 8h00 (Europe/Paris) | `_run_weekly_job()` → `send_weekly_reports()` | PDF + SendGrid pour chaque utilisateur |
| Messages fidélité | `loyalty_messages` | 1er du mois 10h00 | `send_loyalty_messages()` | Gmail SMTP pour les Champions |

**Tolérance misfire :** 3600 secondes (1 heure). Si l'application était éteinte au moment du déclenchement, le job s'exécutera dans l'heure suivant le redémarrage.

**Fuseau horaire :** `Europe/Paris` (CET/CEST selon la saison).

**Thread-safety :** Le scheduler est protégé par `threading.Lock` pour éviter les initialisations multiples lors des reruns Streamlit. Le flag `_scheduler_started` en session state empêche les appels redondants à `start_scheduler()`.

---

## 18. Sécurité et Garde-Fous

Cette section documente l'ensemble des mécanismes de contrôle d'intégrité mis en œuvre dans RetainIQ pour prévenir les erreurs opérationnelles, les données aberrantes et les déclenchements non maîtrisés.

### 18.1 Recalcul Dynamique du Total Cumulé

Chaque déclenchement de campagne est précédé d'une estimation du coût total en temps réel. Ce calcul est effectué côté serveur avant le rendu du bouton, ce qui empêche tout déclenchement sur la base d'une estimation périmée.

**Formule de calcul selon le type de récompense :**

```
Pourcentage %  → coût_unitaire = mean(MonthlyCharges.clip(lower=0)) × reward_value / 100
                  total_cumulé  = coût_unitaire × nb_clients_ciblés

Montant fixe € → coût_unitaire = reward_value
                  total_cumulé  = reward_value × nb_clients_ciblés

En nature       → total_cumulé  = None  (non quantifiable monétairement)
```

Le `.clip(lower=0)` sur `MonthlyCharges` est un garde-fou contre les valeurs négatives aberrantes qui pourraient fausser la moyenne et sous-estimer le coût réel.

**Comparaison au budget configurable :**

```python
depasse_budget = (
    total_cumule is not None
    and budget_max > 0
    and total_cumule > budget_max
)
```

Si `depasse_budget is True` → alerte visuelle (texte rouge) + **bouton de déclenchement désactivé** (`disabled=True`).

### 18.2 Blocage des Campagnes Vides

Le déclenchement d'une campagne sur une liste vide de clients (résultant de filtres trop restrictifs) est bloqué au niveau du composant Streamlit :

```python
campagne_bloquee = (nb_filtres == 0) or depasse_budget

st.button(
    "🚀 Déclencher la campagne",
    disabled=campagne_bloquee,   # désactivé si vide OU budget dépassé
    ...
)
```

Lorsque la campagne est bloquée pour cause de liste vide, un message informatif contextuel guide l'utilisateur vers un ajustement de ses filtres, plutôt qu'un simple message d'erreur.

### 18.3 Score de Qualité des Données (Pipeline ML)

L'entraînement du modèle est **bloqué programmatiquement** (`return` anticipé) si le score de qualité du dataset est inférieur à 30/100 :

```python
if score < 30:
    st.error("❌ La qualité des données est insuffisante (score < 30). ...")
    return   # Blocage : aucun entraînement possible
```

Ce seuil protège contre les modèles entraînés sur des données manifestement corrompues (colonne cible manquante, dataset de moins de 200 lignes, ou déséquilibre de classes extrême).

### 18.4 Validation de la Colonne Cible

Si `detect_columns()` ne trouve aucune colonne cible, le pipeline déclenche un **fallback interactif** au lieu de bloquer :

```python
if not detection["target_col"]:
    binary_cols = [col for col in df_raw.columns if df_raw[col].dropna().nunique() == 2]
    if not binary_cols:
        st.error("❌ Votre fichier ne contient aucune donnée binaire (ex: 0/1, Oui/Non).")
        return
    st.warning("⚠️ Aucune colonne cible n'a pu être détectée automatiquement.")
    chosen_col = st.selectbox("🎯 Quelle colonne indique le départ du client ?", binary_cols)
    df_raw = df_raw.rename(columns={chosen_col: "Churn"})
    detection = detect_columns(df_raw, secteur)   # relance avec la colonne renommée
```

Ce garde-fou permet de traiter n'importe quel CSV dont la colonne cible est nommée hors-nomenclature, sans rejeter le dataset.

### 18.5 Sécurité de l'Authentification

| Mécanisme | Implémentation |
|-----------|---------------|
| Stockage des mots de passe | bcrypt avec sel aléatoire (`bcrypt.gensalt()`) — jamais en clair |
| Migration transparente | SHA256 → bcrypt à la première connexion réussie, sans intervention utilisateur |
| Isolation des données | Chaque modèle et dataset est nommé `model_[email_safe].pkl` — isolation par utilisateur |
| Session Streamlit | `logged_in = False` → arrêt immédiat (`st.stop()`) avant tout affichage |
| WAL SQLite | `PRAGMA journal_mode=WAL` — protection contre les corruptions en écriture concurrente |

### 18.6 Robustesse des Jobs Planifiés

| Risque | Garde-fou |
|--------|-----------|
| Job déclenché alors que l'app était éteinte | Tolérance misfire de 3600 s — le job s'exécute au redémarrage |
| Modèle absent pour un utilisateur | `load_user_model()` retourne `(None, None, None)` → skip silencieux |
| Gmail non configuré | `_send_gratitude_email()` simule le succès et logue en console — pas d'exception levée |
| SendGrid non configuré | Fallback automatique vers `reports_archive/` via `_save_pdf_locally()` |
| Données vides pour un utilisateur | `if df is None` → `continue` dans la boucle du job |

---

## 19. Limites connues et pistes d'amélioration

### Limites actuelles

- **Aucun admin UI :** La gestion des utilisateurs nécessite des requêtes SQLite directes (`sqlite3 retainiq.db`)
- **Fichiers modèles non chiffrés :** Les `model_[email].pkl` sont en clair sur le filesystem
- **Messages de fidélité envoyés à l'email de l'entreprise :** En production, ils devraient être envoyés aux emails des clients finaux
- **Chatbot basé sur règles :** Pas de LLM — réponses limitées aux mots-clés prédéfinis
- **SHAP en mémoire :** Le TreeExplainer charge tout le dataset en RAM (problème si dataset > 100 000 lignes)
- **`loyalty_settings.json` :** Fichier JSON plat partagé — non adapté à un déploiement multi-utilisateurs à haute concurrence (les `reward_primitives` elles sont déjà en SQLite)
- **Pas de multitenancy strict :** Les fichiers modèles sont nommés par email mais dans le répertoire courant
- **Pas de HTTPS natif :** Streamlit Cloud ou reverse proxy (nginx) requis en production
- **Webhook sans authentification :** Le POST JSON vers le webhook n'inclut pas de signature HMAC — à sécuriser en production

### Pistes d'amélioration (v3.0)

- Intégration d'un LLM (Claude API via Anthropic SDK) pour le chatbot contextuel avec mémoire de conversation
- Base de données clients réelle (PostgreSQL) pour les emails des clients finaux
- Interface admin pour la gestion des utilisateurs et des modèles
- Chiffrement des fichiers `.pkl` au repos (Fernet / AES-256)
- Tests unitaires et d'intégration (pytest) avec CI/CD GitHub Actions
- Déploiement Docker avec Traefik pour le HTTPS
- Alertes Slack/Teams en plus des emails via webhooks
- Dashboard d'A/B testing pour les campagnes de rétention
- Intégration CRM (Salesforce, HubSpot) via API REST — branchement natif du payload JSON simulé
- API REST (FastAPI) pour découpler le frontend Streamlit du backend ML
- Sécurisation du webhook (signature HMAC-SHA256, liste blanche d'IPs) pour la mise en production
- Export Excel enrichi avec graphiques intégrés dans la feuille (xlsxwriter `add_chart()`)

---

> **RetainIQ v2.1** — Projet Industriel 2024-2025
> Stack : XGBoost · Streamlit · APScheduler · SHAP · SendGrid · Gmail SMTP · SQLite · bcrypt · ReportLab · xlsxwriter
> Architecture : IA Prédictive + IA Prescriptive · API-First · Webhook HTTP · Catalogue Dynamique · Multi-secteur · Multi-tenant
> *De la prédiction du churn à l'action en boucle fermée — en un seul déploiement.*
