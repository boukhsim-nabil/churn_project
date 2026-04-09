# RetainIQ v2.0 — Plateforme IA de Prédiction & Rétention du Churn

> Projet Industriel 2024-2025 · XGBoost · Streamlit · APScheduler · SHAP · SendGrid · Gmail SMTP · SQLite · bcrypt

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Nouveautés de la v2.0](#2-nouveautés-de-la-v20)
3. [Architecture générale](#3-architecture-générale)
4. [Stack technique](#4-stack-technique)
5. [Structure des fichiers](#5-structure-des-fichiers)
6. [Schéma de la base de données](#6-schéma-de-la-base-de-données)
7. [Installation et démarrage](#7-installation-et-démarrage)
8. [Variables d'environnement](#8-variables-denvironnement)
9. [Workflow complet — étape par étape (v2.0)](#9-workflow-complet--étape-par-étape-v20)
   - 9.1 [Authentification et inscription](#91-authentification-et-inscription)
   - 9.2 [Blank Slate — premier démarrage](#92-blank-slate--premier-démarrage)
   - 9.3 [Import CSV et pipeline de données](#93-import-csv-et-pipeline-de-données)
   - 9.4 [Entraînement du modèle XGBoost](#94-entraînement-du-modèle-xgboost)
   - 9.5 [Dashboard Overview et Visual Analytics](#95-dashboard-overview-et-visual-analytics)
   - 9.6 [Prédiction IA et jauge de risque](#96-prédiction-ia-et-jauge-de-risque)
   - 9.7 [Simulateur What-If](#97-simulateur-what-if)
   - 9.8 [Alertes Clients](#98-alertes-clients)
   - 9.9 [Explainabilité SHAP](#99-explainabilité-shap)
   - 9.10 [Assistant IA (Chatbot)](#910-assistant-ia-chatbot)
   - 9.11 [Programme de Fidélité (NOUVEAU v2.0)](#911-programme-de-fidélité-nouveau-v20)
   - 9.12 [Rapports planifiés et scheduler](#912-rapports-planifiés-et-scheduler)
10. [Documentation complète des modules](#10-documentation-complète-des-modules)
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
11. [Pages du dashboard (toutes)](#11-pages-du-dashboard-toutes)
12. [Système de segmentation clients v2.0](#12-système-de-segmentation-clients-v20)
13. [Catalogue de récompenses par secteur](#13-catalogue-de-récompenses-par-secteur)
14. [Scheduler — deux jobs automatiques](#14-scheduler--deux-jobs-automatiques)
15. [Limites connues et pistes d'amélioration](#15-limites-connues-et-pistes-damélioration)

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
| Page Alertes Clients avec export CSV | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Assistant IA Chatbot contextuel | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Jauge de risque visuelle (Plotly Indicator) | **NOUVEAU** | `churn_prediction_dashboard.py` |
| Migration auth SHA256 → bcrypt transparente | **AMÉLIORÉ** | `auth.py`, `database.py` |
| Navigation conditionnelle (Blank Slate vs modèle actif) | **AMÉLIORÉ** | `churn_prediction_dashboard.py` |
| Fallback local pour rapports PDF (si SendGrid absent) | **AMÉLIORÉ** | `email_reports.py` |
| Fuseau horaire Europe/Paris pour le scheduler | **AMÉLIORÉ** | `scheduler.py` |

---

## 3. Architecture générale

```
churn_prediction_dashboard.py  ← Point d'entrée Streamlit (routeur de pages)
│
├── auth.py                    ← Authentification (login / inscription / migration)
│   └── database.py            ← Couche SQLite (CRUD utilisateurs, sans dépendances)
│
├── data_pipeline.py           ← Import CSV · détection colonnes · nettoyage · XGBoost
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
└── retainiq.db                ← Base SQLite (table users)
```

**Flux de données principal :**

```
[Utilisateur] → auth.py → SQLite
     ↓
[CSV upload] → data_pipeline.py
     ├── detect_columns()     → classification automatique des colonnes
     ├── clean_data()         → imputation · encodage one-hot · normalisation cible
     ├── quality_report()     → score 0-100, avertissements, recommandations
     └── train_custom_model() → XGBoost · sauvegarde model_[email].pkl + data_[email].csv
     ↓
[Dashboard] → prédictions ChurnProba · RiskLevel
     ├── shap_explainer.py    → 4 vues SHAP
     ├── loyalty_page.py      → 3 cohortes + récompenses
     └── email_reports.py     → PDF → SendGrid ou fallback local
     ↓
[APScheduler — 2 jobs]
     ├── Lundi 8h    → weekly_report_job.py → PDF → SendGrid
     └── 1er mois 10h → loyalty_messages_job.py → Gmail SMTP
```

---

## 4. Stack technique

| Couche | Technologie | Version recommandée |
|--------|------------|---------------------|
| Framework web | Streamlit | 1.35+ |
| Machine Learning | XGBoost | 2.0+ |
| Prétraitement | scikit-learn | 1.4+ |
| Manipulation de données | Pandas, NumPy | — |
| Explainabilité | SHAP | — |
| Visualisation | Plotly, Matplotlib, Seaborn | — |
| Base de données | SQLite (via `sqlite3` stdlib) | — |
| Auth / Hashing | bcrypt | — |
| Génération PDF | ReportLab | — |
| Email (rapports) | SendGrid API | — |
| Email (fidélité) | Gmail SMTP (`smtplib`) | — |
| Scheduling | APScheduler (BackgroundScheduler) | 3.x |
| Variables d'env | python-dotenv | — |

---

## 5. Structure des fichiers

```
AI-Powered-Churn-Prediction-main/
│
├── churn_prediction_dashboard.py   # ~1000 lignes — routeur Streamlit, toutes les pages
├── auth.py                         # Authentification, inscription, migration bcrypt
├── database.py                     # Couche SQLite (CRUD) — aucune dépendance projet
├── data_pipeline.py                # Pipeline ML : détection, nettoyage, entraînement
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

## 7. Installation et démarrage

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
sqlite3 retainiq.db ".schema users"
sqlite3 retainiq.db "SELECT email, company, secteur FROM users;"

# Tester le job fidélité en standalone
python loyalty_messages_job.py

# Tester le job hebdomadaire en standalone
python weekly_report_job.py

# Migrer les anciens utilisateurs SHA256
python migrate_users.py
```

---

## 8. Variables d'environnement

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

## 9. Workflow complet — étape par étape (v2.0)

### 9.1 Authentification et inscription

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

### 9.2 Blank Slate — premier démarrage

**Fichier :** `churn_prediction_dashboard.py`

Après connexion, le système vérifie l'existence du fichier `model_[email_safe].pkl` :

- **Sans modèle (Blank Slate) :** navigation réduite à `["🏠 Bienvenue", "📤 Importer mes données"]`. Un bandeau d'avertissement s'affiche dans la sidebar. La page Bienvenue guide l'utilisateur en 3 étapes visuelles.
- **Avec modèle actif :** navigation complète avec 11 pages. La sidebar affiche `✅ Modèle personnalisé actif`.

### 9.3 Import CSV et pipeline de données

**Fichier :** `data_pipeline.py` — Fonction principale : `show_pipeline_page(user_email, secteur)`

Le pipeline se déroule en **6 étapes** visuelles :

**Étape 1 — Upload du fichier CSV**
- Widget `st.file_uploader` acceptant les `.csv`
- Affichage d'un exemple de format si aucun fichier
- Lecture via `pd.read_csv()`

**Étape 2 — Détection automatique des colonnes** (`detect_columns(df, secteur)`)
- Recherche de la colonne cible par `target_hints` spécifiques au secteur
- Fallback : colonne binaire (0/1, Yes/No, Oui/Non, True/False) si hints non trouvées
- Classification : colonnes numériques, catégorielles, ignorées (ID, dates, emails, noms)
- Colonnes catégorielles avec >20 valeurs uniques → ignorées avec avertissement

**Étape 3 — Nettoyage automatique** (`clean_data(df, detection_report)`)
- Colonnes ignorées supprimées
- Numériques : conversion via `pd.to_numeric(errors="coerce")` + imputation par médiane
- Catégorielles : imputation par mode + encodage one-hot (`pd.get_dummies`)
- Colonne cible : mapping Yes/No/Oui/Non/True/False/churned/active → 1/0
- Renommage uniforme de la cible en `"Churn"`
- Suppression des lignes avec cible manquante

**Étape 4 — Rapport de qualité** (`quality_report(df_raw, df_clean, detection_report)`)
- Score de qualité de 0 à 100 (pénalités : déséquilibre, valeurs manquantes, dataset trop petit)
- Taux de churn, répartition des classes, recommandations SMOTE si nécessaire
- Blocage de l'entraînement si score < 30

**Étape 5 — Visualisation avant entraînement**
- Histogrammes Plotly des 2 premières colonnes numériques segmentés par `Churn`

**Étape 6 — Entraînement** (`train_custom_model(df_clean, user_email)`)
- Split 80/20 stratifié (random_state=42)
- XGBoost avec `scale_pos_weight` calculé automatiquement (gestion déséquilibre)
- Sauvegarde `model_[email_safe].pkl` et `data_[email_safe].csv`
- Affichage Accuracy, F1-Score, AUC-ROC avec `st.balloons()`

### 9.4 Entraînement du modèle XGBoost

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

### 9.5 Dashboard Overview et Visual Analytics

**Fichier :** `churn_prediction_dashboard.py`

- **Overview :** 4 KPIs (Total Clients, Taux Churn, Précision Modèle, Clients Urgents), tableau des 10 premiers clients avec score, export CSV complet
- **Visual Analytics :** Pie rétention/churn, histogramme MonthlyCharges vs Churn, boxplot Tenure vs Churn, top 10 features importance XGBoost, distribution churn (Seaborn), histogramme Tenure (Seaborn), matrice de corrélation

Si le modèle custom de l'utilisateur est disponible (`load_user_model()`), il remplace le modèle démo Telco. Le fallback (données aléatoires) s'active si `Telco-Customer-Churn.csv` est absent.

### 9.6 Prédiction IA et jauge de risque

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

### 9.7 Simulateur What-If

**Fichier :** `churn_prediction_dashboard.py` — Page `⚡ Simulateur What-If`

Permet de comparer deux situations (avant/après une action commerciale) :

1. L'utilisateur configure la **situation actuelle** (tenure, charges, contrat, internet, sécurité)
2. Il configure la **situation cible** (après remise, changement de contrat, etc.)
3. `build_input_df()` construit un DataFrame aligné sur `feature_names` pour chaque situation
4. `model.predict_proba()` calcule `score_a` et `score_b`
5. Affichage côte à côte : jauge avant, delta (+ ou −), jauge après
6. Recommandations via `get_recommendations(score, tenure, charges)`

### 9.8 Alertes Clients

**Fichier :** `churn_prediction_dashboard.py` — Page `🚨 Alertes Clients`

- Slider de seuil (30%–90%) pour filtrer `df[df['ChurnProba'] > seuil]`
- Tri par score décroissant, charges décroissantes, ou ancienneté croissante
- KPIs : nombre de clients à risque, score moyen, revenu mensuel menacé
- Tableau paginé des 50 premiers + export CSV
- Bloc envoi manuel de rapport PDF via SendGrid (avec pré-remplissage email utilisateur)
- 3 fiches d'actions recommandées (appel, offre, email)

### 9.9 Explainabilité SHAP

**Fichier :** `shap_explainer.py` — `show_shap_page(model, df, feature_names)`

**Vue 1 — Importance globale :** Barres horizontales des 15 features avec le plus grand `|SHAP moyen|`

**Vue 2 — Impact positif vs négatif :** Barres rouges (augmentent le churn) et vertes (le réduisent), avec ligne centrale à zéro

**Vue 3 — Explication individuelle :** Sélecteur de client (0–100), graphique waterfall SHAP, explication en langage naturel générée par `get_shap_explanation_text()`, profil complet du client

**Vue 4 — Scatter charges vs SHAP :** Nuage de points MonthlyCharges × Impact SHAP, coloré par score de risque

**Optimisation :** `compute_shap_values()` est décorée `@st.cache_data` pour éviter le recalcul à chaque interaction.

### 9.10 Assistant IA (Chatbot)

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

### 9.11 Programme de Fidélité (NOUVEAU v2.0)

**Fichiers :** `loyalty_page.py`, `loyalty_config.py`, `loyalty_messages_job.py`

C'est la grande nouveauté de la v2.0 : la plateforme passe de l'IA **prédictive** à l'IA **prescriptive**.

#### Segmentation en 3 cohortes (`segment_clients(df, secteur)`)

| Cohorte | Critères | Sous-labels | Action |
|---------|----------|-------------|--------|
| 🚨 **Cohorte A — Sauvetage** | `ChurnProba > 0.50` | 🔴 Critique / 🟠 Urgent / 🟡 À suivre | Récompenses de sauvetage immédiates |
| 🏆 **Cohorte B — Fidélité** | `ChurnProba < 0.35` ET `tenure ≥ Q75(tenure)` | 🥇 Légende / 🥈 Vétéran / 🥉 Fidèle | Récompenses de fidélité |
| 🌟 **Champions** | `ChurnProba < 0.20` ET `tenure ≥ 12 mois` | Mur des Champions | Messages automatiques mensuels |

**KPIs globaux :**
- Nombre de clients par cohorte
- Anniversaires de contrat ce mois (`tenure % 12 == 0`)
- Revenu mensuel menacé (somme `MonthlyCharges` cohorte A)

#### Onglets de la page Fidélité

**Tab A — Sauvetage :**
- Filtres par priorité (Critique / Urgent / À suivre)
- Tri par score ou charges
- Sélecteur de récompense depuis `REWARDS_CATALOG[secteur]["sauvetage"]`
- Bouton "Déclencher la campagne"
- Export CSV cohorte A

**Tab B — Fidélité :**
- Liste triée par ancienneté décroissante
- Médailles automatiques (Légende / Vétéran / Fidèle)
- Sélecteur de récompense depuis `REWARDS_CATALOG[secteur]["fidelite"]`
- Bouton "Envoyer aux clients fidèles"
- Export CSV cohorte B

**Tab Champions — Mur des Champions :**
- Grille de 10 cartes visuelles (médaille, ancienneté, score, charges)
- Section "Anniversaires de contrat ce mois"
- Aperçu des messages automatiques (anniversaire et reconnaissance mensuelle)

**Tab Config — Panneau d'administration :**

5 blocs de configuration sauvegardés par utilisateur dans `loyalty_settings.json` :

| Bloc | Paramètres |
|------|------------|
| 🎛️ Bloc 1 — Seuils | Seuil d'urgence (%), Ancienneté min Cohorte B (mois), Dépense minimum (MAD) |
| 💎 Bloc 2 — Valeur | Type de récompense (%, MAD fixe, nature), Valeur numérique |
| 📝 Bloc 3 — Templates | Objet email/SMS (balises dynamiques), Corps du message, Aperçu live |
| 🔒 Bloc 4 — Garde-fous | Budget max/mois (MAD), Quota mensuel (clients), Période de carence (mois) |
| 🔌 Bloc 5 — Activation | Toggle campagne Sauvetage, Toggle campagne Fidélité |

**Balises dynamiques disponibles :** `{client_nom}`, `{anciennete_mois}`, `{valeur_recompense}`, `{score_risque}`, `{nom_entreprise}`

### 9.12 Rapports planifiés et scheduler

**Fichier :** `scheduler.py`

Le scheduler est un **singleton APScheduler** initialisé une seule fois par processus Python (thread-safe via `threading.Lock`).

**Configuration de la planification** (interface Streamlit, page `⏰ Rapports Planifiés`) :
- Sélection du jour (Lundi→Dimanche)
- Heure et minute (0–23, 0–59)
- Bouton "Enregistrer" → `update_schedule()` sans redémarrage
- Bouton "Déclencher maintenant" → `trigger_now()` pour test

**Status affiché :**
- Scheduler actif/inactif (vert/rouge)
- Prochaine exécution prévue
- Nombre de jobs enregistrés
- Historique des 10 dernières exécutions (date, statut, durée)

---

## 10. Documentation complète des modules

### `churn_prediction_dashboard.py`

Point d'entrée Streamlit (~1000 lignes). Gère la session, le CSS global, la navigation et le routage vers toutes les pages.

| Fonction / Section | Description |
|--------------------|-------------|
| `load_data()` | `@st.cache_data` — Charge `Telco-Customer-Churn.csv` ou génère 1000 clients synthétiques si absent |
| `train_model(df)` | `@st.cache_resource` — Entraîne XGBoost sur les données démo |
| `risk_gauge(score)` | Retourne `(fig, label, color, css_class)` — jauge Plotly Indicator tricolore |
| `get_recommendations(score, tenure, charges)` | Retourne 3 recommandations contextuelles selon le niveau de risque |
| `build_input_df(tenure, charges, contract, internet, security)` | Construit un DataFrame aligné sur `feature_names` pour le simulateur What-If |
| `chatbot_response(question)` | Moteur de réponses basé sur mots-clés — retourne une string markdown |
| Section `🏠 Bienvenue` | Page Blank Slate (sans modèle) — 3 cartes étapes + bouton navigation |
| Section `🏠 Overview` | KPIs + tableau 10 clients + export CSV + info dataset |
| Section `📊 Visual Analytics` | 7 graphiques Plotly/Matplotlib/Seaborn |
| Section `🔮 AI Prediction` | Prédiction manuelle/auto + jauge + recommandations |
| Section `🌟 Future Scenarios` | Simulation impact prix/tenure sur la distribution de risque |
| Section `⚡ Simulateur What-If` | Double jauge avant/après + delta + recommandations |
| Section `🚨 Alertes Clients` | Tableau filtrable + export CSV + envoi rapport PDF |
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
| `init_db` | `()` | `CREATE TABLE IF NOT EXISTS users` — idempotent, appelé à l'import |
| `get_user` | `(email: str) → dict | None` | `SELECT * FROM users WHERE email = ?` — retourne dict ou None |
| `create_user` | `(email, password_hash, company, secteur, hash_type) → None` | `INSERT INTO users` — lève `ValueError` si email déjà utilisé |
| `update_user_hash` | `(email, new_hash, new_type) → None` | `UPDATE users SET password_hash, hash_type` — migration bcrypt |
| `get_all_users` | `() → dict` | Retourne tous les users sous forme `{email: {company, secteur, created_at}}` |
| `user_exists` | `(email: str) → bool` | Alias de `get_user(email) is not None` |

---

### `data_pipeline.py`

Pipeline ML complet : de l'upload CSV à l'entraînement XGBoost. Contient aussi `show_pipeline_page()` qui orchestre les 6 étapes Streamlit.

#### Constante `SECTEUR_COLUMNS`

Dictionnaire de configuration par secteur définissant `required` (colonnes requises), `target_hints` (noms possibles de la colonne cible), `description` (texte d'aide).

#### Fonctions

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `detect_columns` | `(df, secteur) → dict` | Détecte `target_col`, `numeric_cols`, `categorical_cols`, `ignored_cols`, `warnings`. Recherche d'abord par `target_hints`, puis fallback binaire. |
| `clean_data` | `(df, detection_report) → (df_clean, cleaning_log)` | Supprime colonnes ignorées, impute numériques (médiane), encode catégorielles (one-hot), mappe cible (Yes/No → 1/0), renomme en `"Churn"`. |
| `quality_report` | `(df_raw, df_clean, detection_report) → dict` | Score 0-100 avec pénalités : churn < 5% (−20), churn > 60% (−15), missing > 20% (−20), missing > 5% (−5), < 200 lignes (−30), < 500 lignes (−10), pas de cible (−40). |
| `train_custom_model` | `(df_clean, user_email) → (model, metrics, error)` | Split stratifié 80/20, XGBoost avec `scale_pos_weight` auto, sauvegarde `.pkl` et `.csv`. Retourne `(None, None, msg_erreur)` si problème. |
| `load_user_model` | `(user_email) → (model, features, df)` | Charge `model_[safe].pkl` et `data_[safe].csv`. Retourne `(None, None, None)` si absent. |
| `show_pipeline_page` | `(user_email, secteur)` | Page Streamlit complète — orchestre les 6 étapes avec UI progressive. |

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
| `start_scheduler` | `(day_of_week, hour, minute) → BackgroundScheduler` | Démarre le scheduler si non actif. Enregistre 2 jobs : `weekly_report` (cron lundi 8h) et `loyalty_messages` (cron 1er du mois 10h). Fuseau : Europe/Paris. Tolérance misfire : 1h. |
| `stop_scheduler` | `()` | Arrête proprement via `shutdown(wait=False)` |
| `get_status` | `() → dict` | Retourne `{running, next_run, job_count, history[:10]}` |
| `trigger_now` | `()` | Lance `_run_weekly_job()` immédiatement (test manuel depuis le dashboard) |
| `update_schedule` | `(day_of_week, hour, minute)` | `reschedule_job("weekly_report", ...)` sans redémarrage. Démarre si inactif. |
| `_refresh_next_run` | `()` | Met à jour `next_run_time` depuis `_scheduler.get_job("weekly_report")` |

**Initialisation dans le dashboard :**

```python
if not st.session_state.get("_scheduler_started"):
    sched.start_scheduler()
    st.session_state["_scheduler_started"] = True
```

---

### `loyalty_page.py`

Page Streamlit complète du Programme de Fidélité (~650 lignes). Gère segmentation, affichage des cohortes, catalogue récompenses, configuration.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `_load_settings` | `(user_email) → dict` | Charge depuis `loyalty_settings.json` les paramètres de l'utilisateur. Merge avec `_DEFAULT_SETTINGS` pour les clés manquantes. |
| `_save_settings` | `(user_email, settings) → None` | Sauvegarde dans `loyalty_settings.json` au format `{email: settings}`. Lecture/écriture complète du fichier. |
| `segment_clients` | `(df, secteur) → (cohorte_a, cohorte_b, champions)` | Retourne 3 DataFrames filtrés selon les seuils de `SEGMENTATION_CONFIG`. Ajoute colonne `Priorité` (A) et `Médaille` (B). |
| `show_loyalty_page` | `(df, secteur, user_company, user_email)` | Page principale : toggle activation, KPIs, 2 graphiques, 4 onglets (Sauvetage, Fidélité, Champions, Config). |

**Paramètres par défaut `_DEFAULT_SETTINGS` :**

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `seuil_urgence` | `0.65` | Seuil Cohorte A (65%) |
| `tenure_min_b` | `12` | Ancienneté min Cohorte B |
| `depense_min` | `0.0` | Dépense minimum (MAD) |
| `reward_type` | `"Pourcentage %"` | Type de récompense |
| `reward_value` | `20.0` | Valeur de la récompense |
| `email_subject` | Template avec balise `{client_nom}` | Objet du message |
| `email_body` | Template complet | Corps du message |
| `budget_max` | `5000.0` | Budget mensuel max (MAD) |
| `quota_mois` | `100` | Quota mensuel |
| `periode_carence` | `3` | Mois entre deux récompenses |
| `campagne_sauvetage` | `True` | Activation campagne A |
| `campagne_fidelite` | `True` | Activation campagne B |

---

### `loyalty_config.py`

Fichier de configuration pur — pas de code Streamlit, pas de dépendances.

#### `REWARDS_CATALOG`

Dictionnaire imbriqué `secteur → {sauvetage: [...], fidelite: [...], seuil_sauvetage, seuil_fidelite, tenure_fidelite, devise}`.

Chaque secteur contient 6 à 8 récompenses par type :

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

## 11. Pages du dashboard (toutes)

| Page | Condition d'accès | Description |
|------|-------------------|-------------|
| `🏠 Bienvenue` | Sans modèle (Blank Slate) | Guide d'onboarding en 3 étapes, bouton vers import |
| `📤 Importer mes données` | Toujours disponible | Pipeline CSV complet en 6 étapes |
| `🏠 Overview` | Avec modèle | KPIs + tableau 10 clients + export |
| `📊 Visual Analytics` | Avec modèle | 7 graphiques analytiques |
| `🔮 AI Prediction` | Avec modèle | Prédiction manuelle ou auto avec jauge |
| `🌟 Future Scenarios` | Avec modèle | Simulation impact prix/tenure |
| `⚡ Simulateur What-If` | Avec modèle | Comparaison avant/après action |
| `🚨 Alertes Clients` | Avec modèle | Filtrage clients à risque + export |
| `🤖 Assistant IA` | Avec modèle | Chatbot contextuel |
| `🧠 Explainable AI` | Avec modèle | 4 vues SHAP |
| `⏰ Rapports Planifiés` | Avec modèle | Config scheduler + historique |
| `🏆 Programme de Fidélité` | Avec modèle | 3 cohortes + récompenses + config |

---

## 12. Système de segmentation clients v2.0

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

## 13. Catalogue de récompenses par secteur

Le catalogue est défini dans `loyalty_config.py` et couvre **5 secteurs × 2 types × 6 à 8 récompenses** chacun, soit 60+ récompenses préconfigurées adaptées au **marché marocain** (devise MAD).

Chaque secteur définit également :
- `seuil_sauvetage` : `0.50` (universel)
- `seuil_fidelite` : `0.35` (universel)
- `tenure_fidelite` : ancienneté minimale pour la Cohorte B (6 mois E-commerce/EdTech, 12 mois Sport/SaaS B2B, 18 mois Télécom)
- `devise` : `MAD`

Les messages de gratitude (`GRATITUDE_MESSAGES`) sont personnalisés par secteur — ton "membre" pour le sport, "apprenant" pour EdTech, "partenaire" pour SaaS B2B, "client fidèle" pour E-commerce et Télécom.

---

## 14. Scheduler — deux jobs automatiques

| Job | ID | Déclencheur | Fonction | Description |
|-----|----|-------------|----------|-------------|
| Rapport hebdomadaire | `weekly_report` | Lundi 8h00 (Europe/Paris) | `_run_weekly_job()` → `send_weekly_reports()` | PDF + SendGrid pour chaque utilisateur |
| Messages fidélité | `loyalty_messages` | 1er du mois 10h00 | `send_loyalty_messages()` | Gmail SMTP pour les Champions |

**Tolérance misfire :** 3600 secondes (1 heure). Si l'application était éteinte au moment du déclenchement, le job s'exécutera dans l'heure suivant le redémarrage.

**Fuseau horaire :** `Europe/Paris` (CET/CEST selon la saison).

**Thread-safety :** Le scheduler est protégé par `threading.Lock` pour éviter les initialisations multiples lors des reruns Streamlit. Le flag `_scheduler_started` en session state empêche les appels redondants à `start_scheduler()`.

---

## 15. Limites connues et pistes d'amélioration

### Limites actuelles

- **Aucun admin UI :** La gestion des utilisateurs nécessite des requêtes SQLite directes (`sqlite3 retainiq.db`)
- **Fichiers modèles non chiffrés :** Les `model_[email].pkl` sont en clair sur le filesystem
- **Messages de fidélité envoyés à l'email de l'entreprise :** En production, ils devraient être envoyés aux emails des clients finaux
- **Chatbot basé sur règles :** Pas de LLM — réponses limitées aux mots-clés prédéfinis
- **SHAP en mémoire :** Le TreeExplainer charge tout le dataset en RAM (problème si dataset > 100 000 lignes)
- **loyalty_settings.json :** Fichier JSON plat partagé — non adapté à un déploiement multi-utilisateurs à haute concurrence
- **Pas de multitenancy strict :** Les fichiers modèles sont nommés par email mais dans le répertoire courant
- **Pas de HTTPS natif :** Streamlit Cloud ou reverse proxy (nginx) requis en production

### Pistes d'amélioration (v3.0)

- Intégration d'un LLM (Claude API) pour le chatbot contextuel
- Base de données clients réelle (PostgreSQL) pour les emails des clients finaux
- Interface admin pour la gestion des utilisateurs
- Chiffrement des fichiers `.pkl` au repos
- Tests unitaires et d'intégration (pytest)
- Déploiement Docker avec Traefik pour le HTTPS
- Alertes Slack/Teams en plus des emails
- Dashboard d'A/B testing pour les campagnes de rétention
- Intégration CRM (Salesforce, HubSpot) via API
- API REST (FastAPI) pour découpler le frontend du backend ML

---

> **RetainIQ v2.0** — Projet Industriel 2024-2025
> Stack : XGBoost · Streamlit · APScheduler · SHAP · SendGrid · Gmail SMTP · SQLite · bcrypt · ReportLab
> Architecture : IA Prédictive + IA Prescriptive · Multi-secteur · Multi-tenant · Marché Marocain
