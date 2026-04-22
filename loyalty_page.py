# ═══════════════════════════════════════════════════════════════════════════════
# loyalty_page.py — Page Streamlit "Programme de Fidélité & Rétention"
# RetainIQ · Module Fidélité · Marché Marocain
# ═══════════════════════════════════════════════════════════════════════════════

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from loyalty_config import REWARDS_CATALOG, SEGMENTATION_CONFIG, GRATITUDE_MESSAGES

# ── FICHIER DE PARAMÈTRES PAR ENTREPRISE ─────────────────────────
_SETTINGS_FILE = "loyalty_settings.json"

_DEFAULT_SETTINGS = {
    # Seuils
    "seuil_urgence":       0.65,
    "tenure_min_b":        12,
    "depense_min":         0.0,
    # Valeur récompense
    "reward_type":         "Pourcentage %",
    "reward_value":        20.0,
    # Templates
    "email_subject":       "🎁 Une offre exclusive pour vous, {client_nom} !",
    "email_body":          (
        "Bonjour {client_nom},\n\n"
        "En tant que client fidèle depuis {anciennete_mois} mois, "
        "nous tenons à vous remercier avec une récompense exclusive : "
        "{valeur_recompense}.\n\n"
        "Cette offre est valable 30 jours. Profitez-en !\n\n"
        "Cordialement,\nL'équipe RetainIQ"
    ),
    # Garde-fous
    "budget_max":          5000.0,
    "quota_mois":          100,
    "periode_carence":     3,
    # Activation
    "campagne_sauvetage":  True,
    "campagne_fidelite":   True,
}


def _load_settings(user_email: str) -> dict:
    """Charge les paramètres de l'utilisateur depuis loyalty_settings.json."""
    if not os.path.exists(_SETTINGS_FILE):
        return _DEFAULT_SETTINGS.copy()
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            all_settings = json.load(f)
        user_settings = all_settings.get(user_email, {})
        merged = _DEFAULT_SETTINGS.copy()
        merged.update(user_settings)
        return merged
    except (json.JSONDecodeError, IOError):
        return _DEFAULT_SETTINGS.copy()


def _save_settings(user_email: str, settings: dict) -> None:
    """Sauvegarde les paramètres de l'utilisateur dans loyalty_settings.json."""
    all_settings = {}
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_settings = {}
    all_settings[user_email] = settings
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_settings, f, ensure_ascii=False, indent=2)


# ── SEGMENTATION DES CLIENTS ─────────────────────────────────────────────────
def segment_clients(df, secteur):
    """
    Segmente les clients en 3 cohortes selon le score de risque et l'ancienneté.
    Retourne 3 DataFrames : cohorte_a, cohorte_b, champions
    """
    cfg = SEGMENTATION_CONFIG

    # Récupérer le seuil d'ancienneté pour la fidélité selon le secteur
    catalog        = REWARDS_CATALOG.get(secteur, {})
    tenure_min_b   = catalog.get("tenure_fidelite", 12)
    tenure_q3      = df['tenure'].quantile(0.75) if 'tenure' in df.columns else tenure_min_b

    # 🚨 Cohorte A — Sauvetage Stratégique
    cohorte_a = df[
        (df['ChurnProba'] > cfg["sauvetage_proba_min"])
    ].copy()
    cohorte_a['Priorité'] = cohorte_a['ChurnProba'].apply(
        lambda x: "🔴 Critique"  if x > 0.80 else
                  "🟠 Urgent"    if x > 0.65 else
                  "🟡 À suivre"
    )

    # 🏆 Cohorte B — Fidélité Historique
    cohorte_b = df[
        (df['ChurnProba'] < cfg["fidelite_proba_max"]) &
        (df['tenure']     >= tenure_q3)
    ].copy() if 'tenure' in df.columns else pd.DataFrame()
    if not cohorte_b.empty:
        cohorte_b['Médaille'] = cohorte_b['tenure'].apply(
            lambda x: "🥇 Légende"   if x >= 60 else
                      "🥈 Vétéran"   if x >= 36 else
                      "🥉 Fidèle"
        )

    # 🌟 Champions — Messages automatiques
    champions = df[
        (df['ChurnProba'] < cfg["champion_proba_max"]) &
        (df['tenure']     >= cfg["champion_tenure_min"])
    ].copy() if 'tenure' in df.columns else pd.DataFrame()

    return cohorte_a, cohorte_b, champions


# ── PAGE PRINCIPALE ───────────────────────────────────────────────────────────
def show_loyalty_page(df, secteur, user_company, user_email: str = ""):
    """
    Affiche la page Programme de Fidélité & Rétention — version actionnable.
    """
    from data_pipeline import triage_risque

    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🏆 Programme de Fidélité & Rétention</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>
            Ciblage précis par Motif de Risque — De l'analyse à l'action en un clic
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── VALIDATION DES DONNÉES ───────────────────────────────────────────────
    if df is None or len(df) == 0:
        st.warning("⚠️ Aucune donnée disponible. Importez vos données via 'Importer mes données'.")
        return

    if 'ChurnProba' not in df.columns:
        st.error("❌ Colonne 'ChurnProba' manquante. Lancez d'abord une prédiction IA.")
        return

    # ── ENRICHISSEMENT TRIAGE ────────────────────────────────────────────────
    # On considère comme "à risque" tous les clients avec ChurnProba > 0.40
    clients_risque_raw = df[df['ChurnProba'] > 0.40].copy()

    if clients_risque_raw.empty:
        st.success("✅ Aucun client à risque détecté (seuil : 40%). Excellent taux de rétention !")
        return

    # Appliquer le moteur de triage pour obtenir 'Motif de Risque'
    clients_enrichis = triage_risque(clients_risque_raw, df)

    # Ajouter la colonne Priorité
    def _label_priorite(proba):
        if proba > 0.80:
            return "🔴 Critique (>80%)"
        elif proba > 0.60:
            return "🟠 Urgent (60-80%)"
        else:
            return "🟡 À suivre (40-60%)"

    clients_enrichis['Priorité'] = clients_enrichis['ChurnProba'].apply(_label_priorite)

    # ── KPIs RAPIDES ─────────────────────────────────────────────────────────
    st.markdown("---")
    nb_critique = (clients_enrichis['ChurnProba'] > 0.80).sum()
    nb_urgent   = ((clients_enrichis['ChurnProba'] > 0.60) & (clients_enrichis['ChurnProba'] <= 0.80)).sum()
    nb_suivre   = ((clients_enrichis['ChurnProba'] > 0.40) & (clients_enrichis['ChurnProba'] <= 0.60)).sum()
    revenu_risque = clients_enrichis['MonthlyCharges'].sum() if 'MonthlyCharges' in clients_enrichis.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#EF4444;'>{nb_critique}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🔴 Critique (&gt;80%)</p>
    </div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#F97316;'>{nb_urgent}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🟠 Urgent (60-80%)</p>
    </div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#EAB308;'>{nb_suivre}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🟡 À suivre (40-60%)</p>
    </div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#EF4444;'>{revenu_risque:.0f} €</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>💸 Revenu mensuel menacé</p>
    </div>""", unsafe_allow_html=True)

    # ── FILTRES DE CIBLAGE ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Ciblage de la campagne")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        options_priorite = ["Tous", "🔴 Critique (>80%)", "🟠 Urgent (60-80%)", "🟡 À suivre (40-60%)"]
        filtre_priorite = st.selectbox(
            "Filtrer par priorité",
            options_priorite,
            key="loyalty_filtre_priorite",
        )

    with col_f2:
        motifs_dispo = sorted(clients_enrichis['Motif de Risque'].unique().tolist())
        options_motifs = ["Tous les motifs"] + motifs_dispo
        filtre_motif = st.selectbox(
            "Filtrer par Motif de Risque",
            options_motifs,
            key="loyalty_filtre_motif",
        )

    # ── APPLICATION DES FILTRES ──────────────────────────────────────────────
    df_filtre = clients_enrichis.copy()

    if filtre_priorite != "Tous":
        df_filtre = df_filtre[df_filtre['Priorité'] == filtre_priorite]

    if filtre_motif != "Tous les motifs":
        df_filtre = df_filtre[df_filtre['Motif de Risque'] == filtre_motif]

    # ── TABLEAU DYNAMIQUE ─────────────────────────────────────────────────────
    st.markdown("---")

    nb_filtres = len(df_filtre)
    if nb_filtres == 0:
        st.info("Aucun client ne correspond aux filtres sélectionnés.")
    else:
        st.markdown(
            f"<p style='color:#94A3B8;font-size:0.9rem;margin-bottom:0.5rem;'>"
            f"<strong style='color:#CBD5E1;'>{nb_filtres} client(s)</strong> "
            f"correspondent aux critères de ciblage sélectionnés.</p>",
            unsafe_allow_html=True,
        )

        # Construire les colonnes à afficher
        display_cols = {}
        if 'customerID' in df_filtre.columns:
            display_cols['customerID'] = 'ID Client'
        if 'tenure' in df_filtre.columns:
            display_cols['tenure'] = 'Ancienneté (mois)'
        if 'MonthlyCharges' in df_filtre.columns:
            display_cols['MonthlyCharges'] = 'Charges/mois'
        display_cols['ChurnProba']      = 'Score de Risque'
        display_cols['Priorité']        = 'Priorité'
        display_cols['Motif de Risque'] = 'Motif de Risque'

        cols_present = [c for c in display_cols.keys() if c in df_filtre.columns]
        df_display = df_filtre[cols_present].sort_values('ChurnProba', ascending=False).head(100).copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display['ChurnProba'] = df_display['ChurnProba'].apply(lambda x: f"{x * 100:.1f}%")
        df_display = df_display.rename(columns=display_cols)

        st.dataframe(df_display, use_container_width=True)

        # Export CSV
        csv_export = df_filtre.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Exporter la sélection (CSV)",
            csv_export,
            "clients_cibles.csv",
            "text/csv",
            use_container_width=False,
        )

    # ── ACTION DE FIDÉLISATION ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0D1B2E,#1a1d2e);
                border:1px solid #667eea;border-radius:14px;
                padding:1.2rem 1.5rem;margin-bottom:1.2rem;'>
        <h3 style='color:white;margin:0 0 4px 0;'>🎁 Action de Fidélisation</h3>
        <p style='color:#64748B;margin:0;font-size:0.88rem;'>
            Choisissez une récompense et déclenchez la campagne pour les clients ciblés ci-dessus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    RECOMPENSES_AGNOSTIQUES = [
        "1 mois offert",
        "Surclassement gratuit (Upgrade)",
        "Accès Premium offert pendant 3 mois",
        "Bon de réduction -20%",
    ]

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        selected_reward = st.selectbox(
            "Choisir la récompense à offrir :",
            RECOMPENSES_AGNOSTIQUES,
            key="loyalty_reward_select",
        )
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        trigger = st.button(
            "🚀 Déclencher la campagne",
            key="loyalty_trigger_btn",
            type="primary",
            use_container_width=True,
        )

    if trigger:
        if nb_filtres == 0:
            st.warning("⚠️ Aucun client sélectionné. Ajustez vos filtres avant de déclencher la campagne.")
        else:
            st.success(
                f"✅ Campagne **'{selected_reward}'** déclenchée avec succès pour "
                f"**{nb_filtres} client(s)** ! Les emails ont été mis en file d'attente."
            )
            st.balloons()

    # ── PANNEAU DE CONFIGURATION (repliable) ─────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Configurer les Récompenses & Règles de campagne", expanded=False):
        _render_config_panel(user_email, user_company, secteur)


def _render_config_panel(user_email: str, user_company: str, secteur: str):
    """Panneau d'administration des règles de récompenses."""

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0D1B2E,#1a1d2e);
                border:1px solid #667eea;border-radius:14px;
                padding:1.2rem 1.5rem;margin-bottom:1.5rem;'>
        <h3 style='color:white;margin:0 0 4px 0;'>⚙️ Panneau d'administration — Règles de Récompenses</h3>
        <p style='color:#64748B;margin:0;font-size:0.88rem;'>
            Paramétrez vos campagnes de rétention. Les règles sont sauvegardées par entreprise.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cfg_data = _load_settings(user_email)

    with st.form("loyalty_config_form", border=False):

        st.markdown("### 🎛️ Bloc 1 — Paramétrage des Seuils")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            seuil_urgence = st.slider(
                "🚨 Seuil d'urgence — Cohorte A (%)", 50, 90,
                int(cfg_data["seuil_urgence"] * 100), step=5,
            )
        with col_s2:
            tenure_min_b = st.number_input(
                "🏆 Ancienneté min — Cohorte B (mois)", 1, 120,
                int(cfg_data["tenure_min_b"]), step=1,
            )
        with col_s3:
            depense_min = st.number_input(
                "💰 Dépense minimum requise (€)", 0.0, 99999.0,
                float(cfg_data["depense_min"]), step=50.0,
            )

        st.markdown("---")
        st.markdown("### 💎 Bloc 2 — Personnalisation de la Valeur")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            reward_type = st.selectbox(
                "🎁 Type de récompense",
                ["Pourcentage %", "Montant fixe €", "En nature"],
                index=["Pourcentage %", "Montant fixe €", "En nature"].index(
                    cfg_data.get("reward_type", "Pourcentage %")
                ) if cfg_data.get("reward_type", "Pourcentage %") in ["Pourcentage %", "Montant fixe €", "En nature"] else 0,
            )
        with col_v2:
            reward_value = st.number_input(
                "📊 Valeur de la récompense", 0.0, 99999.0,
                float(cfg_data["reward_value"]), step=5.0,
            )

        st.markdown("---")
        st.markdown("### 🔒 Bloc 3 — Garde-Fous et Limites")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            budget_max = st.number_input(
                "💵 Budget max / mois (€)", 0.0, 9_999_999.0,
                float(cfg_data["budget_max"]), step=500.0,
            )
        with col_g2:
            quota_mois = st.number_input(
                "👥 Quota de récompenses / mois", 0, 100_000,
                int(cfg_data["quota_mois"]), step=10,
            )
        with col_g3:
            periode_carence = st.number_input(
                "⏳ Période de carence (mois)", 0, 24,
                int(cfg_data["periode_carence"]), step=1,
            )

        st.markdown("---")
        col_save, col_reset, _ = st.columns([1, 1, 2])
        submitted = col_save.form_submit_button(
            "💾 Sauvegarder", use_container_width=True, type="primary",
        )
        reset = col_reset.form_submit_button("↩️ Réinitialiser", use_container_width=True)

    if submitted:
        new_settings = {
            "seuil_urgence":      seuil_urgence / 100.0,
            "tenure_min_b":       int(tenure_min_b),
            "depense_min":        float(depense_min),
            "reward_type":        reward_type,
            "reward_value":       float(reward_value),
            "email_subject":      cfg_data["email_subject"],
            "email_body":         cfg_data["email_body"],
            "budget_max":         float(budget_max),
            "quota_mois":         int(quota_mois),
            "periode_carence":    int(periode_carence),
            "campagne_sauvetage": cfg_data.get("campagne_sauvetage", True),
            "campagne_fidelite":  cfg_data.get("campagne_fidelite", True),
        }
        _save_settings(user_email, new_settings)
        st.success("✅ Configuration sauvegardée avec succès !")

    if reset:
        _save_settings(user_email, _DEFAULT_SETTINGS.copy())
        st.info("↩️ Configuration réinitialisée aux valeurs par défaut.")
        st.rerun()


