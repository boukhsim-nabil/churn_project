import pandas as pd
import numpy as np
import streamlit as st
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION PAR SECTEUR
# Colonnes attendues selon le secteur choisi à l'inscription
# ═══════════════════════════════════════════════════════════════
SECTEUR_COLUMNS = {
    "📱 Télécom": {
        "required":   ["tenure", "MonthlyCharges", "TotalCharges"],
        "target_hints": ["Churn", "churn", "CHURN", "resiliation", "Resiliation"],
        "description": "tenure (ancienneté), MonthlyCharges (forfait), TotalCharges (total), Churn (cible)",
    },
    "💪 Salle de Sport": {
        "required":   ["visites_mois", "abonnement_mensuel", "anciennete_mois"],
        "target_hints": ["resiliation", "Resiliation", "churn", "Churn", "depart"],
        "description": "visites_mois, abonnement_mensuel, anciennete_mois, resiliation (cible)",
    },
    "🛍️ E-commerce": {
        "required":   ["nb_commandes", "panier_moyen", "jours_inactif"],
        "target_hints": ["inactif", "Inactif", "churn", "Churn", "depart"],
        "description": "nb_commandes, panier_moyen, jours_inactif, inactif (cible)",
    },
    "🎓 EdTech": {
        "required":   ["cours_termines", "connexions_semaine", "anciennete_mois"],
        "target_hints": ["desinscription", "Desinscription", "churn", "Churn"],
        "description": "cours_termines, connexions_semaine, anciennete_mois, desinscription (cible)",
    },
    "☁️ SaaS B2B": {
        "required":   ["mrr", "nb_utilisateurs", "anciennete_mois"],
        "target_hints": ["resiliation", "Resiliation", "churn", "Churn", "churned"],
        "description": "mrr, nb_utilisateurs, anciennete_mois, resiliation (cible)",
    },
}

# ═══════════════════════════════════════════════════════════════
# ÉTAPE 1 — DÉTECTION AUTOMATIQUE DES COLONNES
# ═══════════════════════════════════════════════════════════════
GLOBAL_TARGET_SYNONYMS = [
    "churn", "resiliation", "résiliation", "exited", "status",
    "churn_label", "depart", "départ", "departed", "attrition",
    "churned", "inactif", "desinscription", "désinscription",
]

# Colonnes CRM/PII — mises de côté avant XGBoost, réintégrées après prédiction
CRM_COLUMN_SYNONYMS = [
    "email", "mail", "courriel", "telephone", "phone", "tel", "mobile",
    "nom", "prenom", "customerid", "client_id", "customer_id", "id_client",
]

def detect_columns(df, secteur):
    """
    Détecte automatiquement :
    - La colonne cible (churn / résiliation)
    - Les colonnes numériques utiles
    - Les colonnes catégorielles
    - Les colonnes à ignorer (ID, dates...)
    """
    report = {
        "target_col":      None,
        "numeric_cols":    [],
        "categorical_cols":[],
        "ignored_cols":    [],
        "warnings":        [],
    }

    # Nettoyer les espaces parasites en tête/queue des noms de colonnes
    df.columns = df.columns.str.strip()

    sector_hints = SECTEUR_COLUMNS.get(secteur, {}).get("target_hints", [])
    all_hints = list(dict.fromkeys([h.lower() for h in sector_hints] + GLOBAL_TARGET_SYNONYMS))

    # Détecter la colonne cible
    for col in df.columns:
        if col.lower() in all_hints:
            report["target_col"] = col
            break

    # Si pas trouvée, chercher une colonne binaire (0/1 ou Yes/No)
    if not report["target_col"]:
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                vals_lower = [str(v).lower() for v in unique_vals]
                if set(vals_lower) <= {"0", "1", "yes", "no", "true", "false", "oui", "non"}:
                    report["target_col"] = col
                    report["warnings"].append(
                        f"Colonne cible auto-détectée : '{col}' — vérifiez que c'est bien votre variable de churn."
                    )
                    break

    # Classifier les autres colonnes
    for col in df.columns:
        if col == report["target_col"]:
            continue

        # Colonnes à ignorer (ID, dates, texte libre)
        if any(kw in col.lower() for kw in ["id", "date", "nom", "name", "email", "phone", "tel"]):
            report["ignored_cols"].append(col)
            continue

        if df[col].dtype in ["int64", "float64"]:
            report["numeric_cols"].append(col)
        elif df[col].dtype == "object":
            if df[col].nunique() <= 20:
                report["categorical_cols"].append(col)
            else:
                report["ignored_cols"].append(col)
                report["warnings"].append(
                    f"Colonne '{col}' ignorée — trop de valeurs uniques ({df[col].nunique()})."
                )

    return report

# ═══════════════════════════════════════════════════════════════
# ÉTAPE 2 — NETTOYAGE AUTOMATIQUE
# ═══════════════════════════════════════════════════════════════
def clean_data(df, detection_report):
    """
    Nettoie automatiquement :
    - Valeurs manquantes
    - Encodage des colonnes catégorielles
    - Conversion des colonnes cibles Yes/No → 1/0
    - Suppression des colonnes inutiles
    """
    cleaning_log = []
    df_clean = df.copy()

    target_col      = detection_report["target_col"]
    numeric_cols    = detection_report["numeric_cols"]
    categorical_cols= detection_report["categorical_cols"]
    ignored_cols    = detection_report["ignored_cols"]

    # Supprimer colonnes ignorées
    cols_to_drop = [c for c in ignored_cols if c in df_clean.columns]
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        cleaning_log.append(f"Supprimé {len(cols_to_drop)} colonne(s) inutiles : {cols_to_drop}")

    # Nettoyer colonnes numériques
    for col in numeric_cols:
        if col not in df_clean.columns:
            continue
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        n_missing = df_clean[col].isna().sum()
        if n_missing > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            cleaning_log.append(f"'{col}' : {n_missing} valeur(s) manquante(s) remplacée(s) par la médiane ({median_val:.2f})")

    # Encoder colonnes catégorielles
    for col in categorical_cols:
        if col not in df_clean.columns:
            continue
        n_missing = df_clean[col].isna().sum()
        if n_missing > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            cleaning_log.append(f"'{col}' : {n_missing} valeur(s) manquante(s) remplacée(s) par le mode ('{mode_val}')")
        df_clean = pd.get_dummies(df_clean, columns=[col], drop_first=True)
        cleaning_log.append(f"'{col}' encodée en one-hot")

    # Nettoyer et encoder la colonne cible
    if target_col and target_col in df_clean.columns:
        # Convertir Yes/No, Oui/Non, True/False → 1/0
        mapping = {
            "yes": 1, "no": 0, "oui": 1, "non": 0,
            "true": 1, "false": 0, "1": 1, "0": 0,
            "churned": 1, "active": 0, "actif": 0
        }
        if df_clean[target_col].dtype == "object":
            df_clean[target_col] = df_clean[target_col].str.lower().map(mapping)
            cleaning_log.append(f"Colonne cible '{target_col}' convertie : Yes/No → 1/0")

        n_missing_target = df_clean[target_col].isna().sum()
        if n_missing_target > 0:
            df_clean.dropna(subset=[target_col], inplace=True)
            cleaning_log.append(f"Supprimé {n_missing_target} ligne(s) avec cible manquante")

        df_clean[target_col] = df_clean[target_col].astype(int)

        # Renommer la colonne cible en "Churn" pour uniformité
        if target_col != "Churn":
            df_clean.rename(columns={target_col: "Churn"}, inplace=True)
            cleaning_log.append(f"Colonne cible renommée '{target_col}' → 'Churn'")

    cleaning_log.append(f"Dataset final : {len(df_clean)} lignes · {len(df_clean.columns)} colonnes")
    return df_clean, cleaning_log

# ═══════════════════════════════════════════════════════════════
# ÉTAPE 3 — RAPPORT DE QUALITÉ
# ═══════════════════════════════════════════════════════════════
def quality_report(df_raw, df_clean, detection_report):
    """
    Génère un rapport de qualité complet des données
    """
    report = {
        "n_rows_original":  len(df_raw),
        "n_rows_clean":     len(df_clean),
        "n_cols_original":  len(df_raw.columns),
        "n_cols_clean":     len(df_clean.columns),
        "n_missing_total":  df_raw.isnull().sum().sum(),
        "pct_missing":      (df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100,
        "target_col":       detection_report["target_col"],
        "churn_rate":       None,
        "class_balance":    None,
        "score":            100,
        "issues":           [],
        "recommendations":  [],
    }

    # Taux de churn
    if "Churn" in df_clean.columns:
        churn_rate = df_clean["Churn"].mean() * 100
        report["churn_rate"]    = churn_rate
        report["class_balance"] = {
            "churned": int(df_clean["Churn"].sum()),
            "active":  int((df_clean["Churn"] == 0).sum())
        }

        # Déséquilibre de classes
        if churn_rate < 5:
            report["issues"].append("Taux de churn très faible (<5%) — données peut-être déséquilibrées")
            report["recommendations"].append("Utiliser SMOTE ou class_weight pour rééquilibrer les classes")
            report["score"] -= 20
        elif churn_rate > 60:
            report["issues"].append("Taux de churn élevé (>60%) — vérifiez vos données")
            report["score"] -= 15

    # Valeurs manquantes
    if report["pct_missing"] > 20:
        report["issues"].append(f"Beaucoup de valeurs manquantes ({report['pct_missing']:.1f}%)")
        report["score"] -= 20
    elif report["pct_missing"] > 5:
        report["issues"].append(f"Quelques valeurs manquantes ({report['pct_missing']:.1f}%) — gérées automatiquement")
        report["score"] -= 5

    # Taille du dataset
    if report["n_rows_original"] < 200:
        report["issues"].append("Dataset trop petit (<200 lignes) — résultats peu fiables")
        report["recommendations"].append("Idéalement 1000+ lignes pour un bon modèle")
        report["score"] -= 30
    elif report["n_rows_original"] < 500:
        report["issues"].append("Dataset petit (<500 lignes) — résultats moyennement fiables")
        report["score"] -= 10

    # Colonne cible manquante
    if not detection_report["target_col"]:
        report["issues"].append("Colonne cible (churn) non détectée")
        report["score"] -= 40

    report["score"] = max(0, report["score"])
    return report

# ═══════════════════════════════════════════════════════════════
# ÉTAPE 4 — ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════
def train_custom_model(df_clean, user_email, crm_df=None):
    """
    Entraîne un modèle XGBoost sur les données nettoyées
    et le sauvegarde pour cet utilisateur spécifiquement.
    crm_df : colonnes PII/CRM à réintégrer dans le CSV sauvegardé (sans passer au modèle).
    """
    if "Churn" not in df_clean.columns:
        return None, None, "Colonne 'Churn' introuvable dans les données nettoyées."

    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]

    if len(df_clean) < 50:
        return None, None, "Pas assez de données pour entraîner un modèle (minimum 50 lignes)."

    # Calcul du poids de classe pour gérer le déséquilibre
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_proba= model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "auc_roc":   round(roc_auc_score(y_test, y_pred_proba) * 100, 2),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "features":  list(X.columns),
        "churn_rate":round(y.mean() * 100, 2),
    }

    # Sauvegarder le modèle pour cet utilisateur
    safe_email  = user_email.replace("@", "_at_").replace(".", "_")
    model_path  = f"model_{safe_email}.pkl"
    data_path   = f"data_{safe_email}.csv"

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)

    # Réintégrer les colonnes CRM dans le CSV sauvegardé (elles ne sont PAS des features ML)
    if crm_df is not None and len(crm_df) == len(df_clean):
        df_to_save = pd.concat(
            [df_clean.reset_index(drop=True), crm_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        df_to_save = df_clean
    df_to_save.to_csv(data_path, index=False)

    return model, metrics, None

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE PERSONNALISÉ
# ═══════════════════════════════════════════════════════════════
def load_user_model(user_email):
    """
    Charge le modèle entraîné pour cet utilisateur.
    Si pas de modèle custom, retourne None.
    """
    safe_email = user_email.replace("@", "_at_").replace(".", "_")
    model_path = f"model_{safe_email}.pkl"
    data_path  = f"data_{safe_email}.csv"

    if not os.path.exists(model_path):
        return None, None, None

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model    = saved["model"]
    features = saved["features"]
    df       = pd.read_csv(data_path) if os.path.exists(data_path) else None

    return model, features, df

# ═══════════════════════════════════════════════════════════════
# MOTEUR DE TRIAGE STATISTIQUE (agnostique au secteur)
# ═══════════════════════════════════════════════════════════════
def triage_risque(df_risque: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les clients à haut risque et attribue un motif + action suggérée.
    Logique entièrement basée sur des seuils statistiques, sans terminologie sectorielle.
    """
    charges_p75 = df_full['MonthlyCharges'].quantile(0.75) if 'MonthlyCharges' in df_full.columns else None
    result = df_risque.copy()

    def _motif_action(row):
        # 1. Détection contrat court — recherche la colonne Contract (brute ou one-hot)
        contrat_mensuel = False
        if 'Contract' in row.index:
            contrat_mensuel = 'month' in str(row['Contract']).lower()
        else:
            one_yr = row.get('Contract_One_year', row.get('Contract_One year', None))
            two_yr = row.get('Contract_Two_year', row.get('Contract_Two year', None))
            if one_yr is not None and two_yr is not None:
                contrat_mensuel = (one_yr == 0) and (two_yr == 0)

        # 2. Pression tarifaire
        pression_tarif = (
            charges_p75 is not None
            and 'MonthlyCharges' in row.index
            and pd.notna(row['MonthlyCharges'])
            and row['MonthlyCharges'] > charges_p75
        )

        # 3. Nouveau client
        nouveau_client = (
            'tenure' in row.index
            and pd.notna(row['tenure'])
            and row['tenure'] <= 6
        )

        if nouveau_client and pression_tarif:
            return pd.Series({
                'Motif de Risque': 'Nouveau client — pression tarifaire',
                'Action Suggérée': 'Offre de bienvenue + options premium à prix réduit',
            })
        if contrat_mensuel:
            return pd.Series({
                'Motif de Risque': "Absence d'engagement",
                'Action Suggérée': 'Proposer un engagement long terme avec avantage tarifaire',
            })
        if pression_tarif:
            return pd.Series({
                'Motif de Risque': 'Pression tarifaire',
                'Action Suggérée': "Audit des services souscrits + offre d'optimisation de coût",
            })
        if nouveau_client:
            return pd.Series({
                'Motif de Risque': 'Nouveau client',
                'Action Suggérée': "Programme d'onboarding renforcé + contact personnalisé J+7",
            })
        return pd.Series({
            "Motif de Risque": "Risque d'insatisfaction globale",
            'Action Suggérée': "Enquête de satisfaction ciblée + appel de rétention sous 48h",
        })

    triage_df = result.apply(_motif_action, axis=1)
    result['Motif de Risque'] = triage_df['Motif de Risque']
    result['Action Suggérée'] = triage_df['Action Suggérée']
    return result


# ═══════════════════════════════════════════════════════════════
# PAGE COMPLÈTE DU PIPELINE (affichée dans Streamlit)
# ═══════════════════════════════════════════════════════════════
def show_pipeline_page(user_email, secteur):
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>📤 Importer vos données</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>
            Uploadez votre fichier CSV → Nettoyage automatique → Entraînement IA en 1 clic
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Info sur le format attendu
    secteur_info = SECTEUR_COLUMNS.get(secteur, {})
    st.info(f"📋 **Format attendu pour {secteur}** : {secteur_info.get('description', 'colonnes numériques + colonne cible churn')}")

    # ── ÉTAPE 1 : Upload ────────────────────────────────────────
    st.subheader("📁 Étape 1 — Uploader votre fichier CSV")
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier CSV ici",
        type=["csv"],
        help="Le fichier doit contenir une colonne de churn (Oui/Non ou 1/0) et des colonnes client."
    )

    if not uploaded_file:
        # Montrer un exemple de format
        st.markdown("#### Exemple de format accepté")
        example_data = {
            "tenure":         [12, 3, 48, 7, 36],
            "MonthlyCharges": [65.5, 89.0, 45.0, 92.0, 55.0],
            "TotalCharges":   [786, 267, 2160, 644, 1980],
            "Contract":       ["Month-to-month", "Month-to-month", "Two year", "Month-to-month", "One year"],
            "Churn":          ["No", "Yes", "No", "Yes", "No"],
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        st.caption("Votre colonne cible peut s'appeler : Churn, churn, resiliation, depart, inactif...")
        return

    # ── LECTURE DU FICHIER ───────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return

    st.success(f"✅ Fichier chargé : **{len(df_raw)} lignes** · **{len(df_raw.columns)} colonnes**")

    # Aperçu brut
    with st.expander("👁️ Aperçu des données brutes", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes",    len(df_raw))
        col2.metric("Colonnes",  len(df_raw.columns))
        col3.metric("Valeurs manquantes", int(df_raw.isnull().sum().sum()))

    # ── ÉTAPE 2 : DÉTECTION DES COLONNES ────────────────────────
    st.markdown("---")
    st.subheader("🔍 Étape 2 — Détection automatique des colonnes")

    # Nettoyage des noms de colonnes (espaces parasites)
    df_raw.columns = df_raw.columns.str.strip()

    # ── Le Douanier (PII) — extraire les colonnes CRM avant tout traitement ML ──
    _crm_cols = [c for c in df_raw.columns if any(kw in c.lower() for kw in CRM_COLUMN_SYNONYMS)]
    _crm_df = df_raw[_crm_cols].copy() if _crm_cols else None

    detection = detect_columns(df_raw, secteur)

    # Fallback interactif si aucune colonne cible détectée
    if not detection["target_col"]:
        binary_cols = [
            col for col in df_raw.columns
            if df_raw[col].dropna().nunique() == 2
        ]
        if not binary_cols:
            st.error(
                "❌ Votre fichier ne contient aucune donnée binaire (ex: 0/1, Oui/Non). "
                "Il est impossible d'entraîner l'IA sans historique de Churn. "
                "Veuillez importer un dataset valide."
            )
            return
        st.warning("⚠️ Aucune colonne cible n'a pu être détectée automatiquement.")
        chosen_col = st.selectbox(
            "🎯 Quelle colonne indique le départ du client ?",
            options=[""] + binary_cols,
            key="manual_target_col_select",
            help="Sélectionnez la colonne qui contient 0/1 ou Oui/Non indiquant si le client a churné.",
        )
        if not chosen_col:
            st.info("Sélectionnez la colonne cible pour continuer le pipeline.")
            return
        df_raw = df_raw.rename(columns={chosen_col: "Churn"})
        detection = detect_columns(df_raw, secteur)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Colonne cible",      detection["target_col"] or "Non trouvée")
    c2.metric("Colonnes numériques", len(detection["numeric_cols"]))
    c3.metric("Colonnes catégoriques", len(detection["categorical_cols"]))
    c4.metric("Colonnes ignorées",  len(detection["ignored_cols"]))

    if detection["warnings"]:
        for w in detection["warnings"]:
            st.warning(f"⚠️ {w}")

    # ── Vérification CRM ────────────────────────────────────────────
    _has_email_col = any(
        any(kw in c.lower() for kw in ["email", "mail", "courriel"])
        for c in df_raw.columns
    )
    if not _has_email_col:
        st.warning(
            "⚠️ Aucune colonne e-mail client détectée dans votre fichier. "
            "Pour activer la relance personnalisée et la synchronisation CRM (Brevo), "
            "ajoutez une colonne `email` ou `mail` à votre prochain import."
        )

    # ── Vérification Performance ML ─────────────────────────────────
    _vital_ml_groups = {
        "ancienneté client":   ["tenure", "anciennete_mois", "mois_inscrit", "mois_client", "ancienneté"],
        "charges mensuelles":  ["monthlycharges", "abonnement_mensuel", "mrr", "panier_moyen", "charges_mensuelles"],
        "type de contrat":     ["contract", "type_contrat", "contrat", "engagement"],
        "tickets support":     ["numtechtickets", "tickets_support", "nb_tickets", "support_calls"],
    }
    _all_cols_lower = [c.lower() for c in df_raw.columns]
    _missing_vital = [
        label for label, syns in _vital_ml_groups.items()
        if not any(s in _all_cols_lower for s in syns)
    ]
    if _missing_vital:
        st.info(
            f"💡 Pour des prédictions plus précises, enrichissez votre prochain CSV avec : "
            f"**{', '.join(_missing_vital)}**."
        )

    with st.expander("📋 Détail des colonnes détectées"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Colonnes numériques utilisées :**")
            for c in detection["numeric_cols"]:
                st.write(f"  ✅ `{c}`")
            st.write("**Colonnes catégorielles (encodées) :**")
            for c in detection["categorical_cols"]:
                st.write(f"  🔄 `{c}`")
        with col2:
            st.write("**Colonne cible :**")
            st.write(f"  🎯 `{detection['target_col']}`")
            st.write("**Colonnes ignorées :**")
            for c in detection["ignored_cols"]:
                st.write(f"  ❌ `{c}`")

    # ── ÉTAPE 3 : NETTOYAGE ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🧹 Étape 3 — Nettoyage automatique")

    df_clean, cleaning_log = clean_data(df_raw, detection)

    # Aligner crm_df sur les lignes conservées par clean_data (ex: suppression target NaN)
    if _crm_df is not None:
        _crm_df = _crm_df.loc[df_clean.index].reset_index(drop=True)

    for log in cleaning_log:
        st.write(f"  ✔️ {log}")

    st.success(f"✅ Données nettoyées : **{len(df_clean)} lignes** · **{len(df_clean.columns)} colonnes**")

    # ── ÉTAPE 4 : RAPPORT DE QUALITÉ ────────────────────────────
    st.markdown("---")
    st.subheader("📊 Étape 4 — Rapport de qualité")

    qr = quality_report(df_raw, df_clean, detection)

    # Score de qualité
    score = qr["score"]
    score_color = "#00CC96" if score >= 80 else "#F59E0B" if score >= 50 else "#EF4444"
    score_label = "Excellent" if score >= 80 else "Correct" if score >= 50 else "Insuffisant"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:{score_color};'>{score}/100</h2>
        <p>Qualité des données — {score_label}</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;'>{qr['churn_rate']:.1f}%</h2>
        <p>Taux de churn dans vos données</p>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;'>{qr['pct_missing']:.1f}%</h2>
        <p>Valeurs manquantes (avant nettoyage)</p>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;'>{qr['n_rows_clean']:,}</h2>
        <p>Lignes utilisables pour l'entraînement</p>
    </div>""", unsafe_allow_html=True)

    # Problèmes et recommandations
    if qr["issues"]:
        st.markdown("**⚠️ Points d'attention :**")
        for issue in qr["issues"]:
            st.warning(issue)

    if qr["recommendations"]:
        st.markdown("**💡 Recommandations :**")
        for rec in qr["recommendations"]:
            st.info(rec)

    # Distribution de la variable cible
    if qr["class_balance"]:
        st.markdown("**Répartition de votre variable cible :**")
        c1, c2 = st.columns(2)
        total = qr["class_balance"]["churned"] + qr["class_balance"]["active"]
        if total == 0:
            st.warning("La colonne cible sélectionnée ne contient pas de valeurs binaires (0/1). Veuillez choisir une colonne Churn valide.")
        else:
            c1.metric("Clients churned (1)", f"{qr['class_balance']['churned']} ({qr['class_balance']['churned']/total*100:.1f}%)")
            c2.metric("Clients actifs (0)",  f"{qr['class_balance']['active']} ({qr['class_balance']['active']/total*100:.1f}%)")

    # Visualisation des données nettoyées
    st.markdown("---")
    st.subheader("📈 Étape 5 — Visualisation avant entraînement")

    numeric_preview = [c for c in df_clean.columns if df_clean[c].dtype in ["int64", "float64"] and c != "Churn"][:4]

    if numeric_preview and "Churn" in df_clean.columns:
        cols = st.columns(min(len(numeric_preview), 2))
        for i, col_name in enumerate(numeric_preview[:2]):
            with cols[i]:
                import plotly.express as px
                fig = px.histogram(
                    df_clean, x=col_name, color="Churn",
                    title=f"{col_name} vs Churn",
                    nbins=25, color_discrete_sequence=["#00CC96", "#EF553B"],
                    barmode="overlay", opacity=0.75
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#CBD5E1",
                    height=280,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── ÉTAPE 6 : ENTRAÎNEMENT ───────────────────────────────────
    st.markdown("---")
    st.subheader("🚀 Étape 6 — Entraîner le modèle IA sur vos données")

    if score < 30:
        st.error("❌ La qualité des données est insuffisante (score < 30). Corrigez les problèmes avant d'entraîner.")
        return

    st.info(f"Le modèle XGBoost va être entraîné sur **{len(df_clean)} clients** avec **{len(df_clean.columns)-1} features**.")

    if st.button("🧠 Lancer l'entraînement du modèle", type="primary", use_container_width=True):
        with st.spinner("Entraînement en cours... Cela peut prendre quelques secondes."):
            model, metrics, error = train_custom_model(df_clean, user_email, crm_df=_crm_df)

        if error:
            st.error(f"❌ Erreur : {error}")
            return

        # Résultats
        st.balloons()
        st.success("✅ Modèle entraîné et sauvegardé avec succès !")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class='metric-container'>
            <h2 style='margin:0;font-size:2.2rem;'>{metrics['accuracy']}%</h2>
            <p>Accuracy</p>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-container'>
            <h2 style='margin:0;font-size:2.2rem;'>{metrics['f1_score']}%</h2>
            <p>F1-Score</p>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-container'>
            <h2 style='margin:0;font-size:2.2rem;'>{metrics['auc_roc']}%</h2>
            <p>AUC-ROC</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='section-card'>
            <p style='color:#CBD5E1;margin:0;'>
                Entraîné sur <b>{metrics['n_train']}</b> clients ·
                Testé sur <b>{metrics['n_test']}</b> clients ·
                <b>{len(metrics['features'])}</b> features utilisées
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.success("✅ Votre modèle est maintenant actif ! Retournez sur le dashboard pour voir les prédictions sur vos données.")

        # Sauvegarder le statut dans la session
        st.session_state["custom_model_trained"] = True
        st.session_state["custom_model_metrics"]  = metrics
