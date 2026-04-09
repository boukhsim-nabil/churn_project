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
    Affiche la page complète Programme de Fidélité & Rétention.
    """

    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🏆 Programme de Fidélité & Rétention</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>
            De l'IA prédictive à l'IA prescriptive — Arsenal anti-churn adapté au marché marocain
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Vérifier que les données sont disponibles
    if df is None or len(df) == 0:
        st.warning("⚠️ Aucune donnée disponible. Importez vos données via 'Importer mes données'.")
        return

    if 'ChurnProba' not in df.columns:
        st.error("❌ Colonne 'ChurnProba' manquante. Lancez d'abord une prédiction IA.")
        return

    if 'tenure' not in df.columns:
        st.error("❌ Colonne 'tenure' (ancienneté) manquante dans vos données.")
        return

    # ── TOGGLE ACTIVATION ───────────────────────────────────────────────────
    col_toggle, col_info = st.columns([1, 3])
    with col_toggle:
        campagnes_actives = st.toggle("🎯 Activer les campagnes", value=True)
    with col_info:
        if campagnes_actives:
            st.success("✅ Les campagnes de récompenses sont **actives**. Les segments sont calculés en temps réel.")
        else:
            st.info("💤 Campagnes désactivées — les segments restent visibles mais aucune action n'est déclenchée.")

    # ── SEGMENTATION ────────────────────────────────────────────────────────
    cohorte_a, cohorte_b, champions = segment_clients(df, secteur)

    # ── KPIs GLOBAUX ─────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#EF4444;'>{len(cohorte_a)}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🚨 Clients à sauver</p>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#02C39A;'>{len(cohorte_b)}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🏆 Clients fidèles</p>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#F59E0B;'>{len(champions)}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>🌟 Champions</p>
    </div>""", unsafe_allow_html=True)

    msg_ce_mois = len(champions[champions['tenure'] % 12 == 0]) if not champions.empty else 0
    c4.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#667eea;'>{msg_ce_mois}</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>📧 Anniversaires ce mois</p>
    </div>""", unsafe_allow_html=True)

    revenu_risque = cohorte_a['MonthlyCharges'].sum() if 'MonthlyCharges' in cohorte_a.columns and not cohorte_a.empty else 0
    c5.markdown(f"""<div class='metric-container'>
        <h2 style='margin:0;font-size:2rem;color:#EF4444;'>{revenu_risque:.0f} MAD</h2>
        <p style='margin:4px 0 0;font-size:0.85rem;'>💸 Revenu mensuel menacé</p>
    </div>""", unsafe_allow_html=True)

    # ── GRAPHIQUE DE SEGMENTATION ────────────────────────────────────────────
    st.markdown("---")
    col_graph, col_dist = st.columns(2)

    with col_graph:
        st.subheader("📊 Distribution des segments")
        seg_data = pd.DataFrame({
            'Segment':  ['🚨 Sauvetage (A)', '🏆 Fidélité (B)', '🌟 Champions', '📊 Autres'],
            'Clients':  [
                len(cohorte_a),
                len(cohorte_b),
                len(champions),
                len(df) - len(cohorte_a) - len(cohorte_b) - len(champions)
            ],
        })
        fig = px.pie(
            seg_data, values='Clients', names='Segment',
            color_discrete_sequence=['#EF4444', '#02C39A', '#F59E0B', '#64748B'],
            hole=0.45
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#CBD5E1', height=300, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        st.subheader("📈 Score de risque vs Ancienneté")
        sample_df = df.sample(min(300, len(df))).copy()
        sample_df['Segment'] = '📊 Autres'
        sample_df.loc[sample_df['ChurnProba'] > 0.50, 'Segment'] = '🚨 Sauvetage'
        sample_df.loc[
            (sample_df['ChurnProba'] < 0.35) & (sample_df['tenure'] >= df['tenure'].quantile(0.75)),
            'Segment'
        ] = '🏆 Fidélité'
        sample_df.loc[
            (sample_df['ChurnProba'] < 0.20) & (sample_df['tenure'] >= 12),
            'Segment'
        ] = '🌟 Champion'

        fig2 = px.scatter(
            sample_df, x='tenure', y='ChurnProba',
            color='Segment',
            color_discrete_map={
                '🚨 Sauvetage': '#EF4444',
                '🏆 Fidélité':  '#02C39A',
                '🌟 Champion':  '#F59E0B',
                '📊 Autres':    '#64748B',
            },
            labels={'tenure': 'Ancienneté (mois)', 'ChurnProba': 'Score de risque'},
            opacity=0.7
        )
        fig2.add_hline(y=0.50, line_dash="dash", line_color="#EF4444",
                       annotation_text="Seuil sauvetage")
        fig2.add_hline(y=0.35, line_dash="dash", line_color="#02C39A",
                       annotation_text="Seuil fidélité")
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#CBD5E1', height=300, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ONGLETS COHORTES ─────────────────────────────────────────────────────
    st.markdown("---")
    tab_a, tab_b, tab_champ, tab_config = st.tabs([
        "🚨 Cohorte A — Sauvetage",
        "🏆 Cohorte B — Fidélité",
        "🌟 Mur des Champions",
        "⚙️ Configurer les Récompenses",
    ])

    # ── TAB A : SAUVETAGE ────────────────────────────────────────────────────
    with tab_a:
        st.subheader(f"🚨 Cohorte A — Sauvetage Stratégique ({len(cohorte_a)} clients)")
        st.info("Ces clients ont un score de churn > 50%. Une action commerciale immédiate peut les retenir.")

        if cohorte_a.empty:
            st.success("✅ Aucun client en zone critique. Excellent travail de rétention !")
        else:
            # Filtres
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filtre_priorite = st.multiselect(
                    "Filtrer par priorité",
                    ["🔴 Critique", "🟠 Urgent", "🟡 À suivre"],
                    default=["🔴 Critique", "🟠 Urgent", "🟡 À suivre"]
                )
            with col_f2:
                tri_a = st.selectbox("Trier par", ["Score décroissant", "Charges décroissantes"])

            df_a_show = cohorte_a[cohorte_a['Priorité'].isin(filtre_priorite)].copy()
            if tri_a == "Score décroissant":
                df_a_show = df_a_show.sort_values('ChurnProba', ascending=False)
            else:
                df_a_show = df_a_show.sort_values('MonthlyCharges', ascending=False) if 'MonthlyCharges' in df_a_show.columns else df_a_show

            # Affichage
            show_cols_a = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnProba', 'Priorité'] if c in df_a_show.columns]
            df_display_a = df_a_show[show_cols_a].head(50).copy()
            df_display_a.index = range(1, len(df_display_a) + 1)
            if 'ChurnProba' in df_display_a.columns:
                df_display_a['ChurnProba'] = df_display_a['ChurnProba'].apply(lambda x: f"{x*100:.1f}%")
            df_display_a = df_display_a.rename(columns={
                'tenure': 'Ancienneté (mois)',
                'MonthlyCharges': 'Charges/mois (MAD)',
                'TotalCharges': 'Total cumulé (MAD)',
                'ChurnProba': '⚠️ Score de risque',
            })
            st.dataframe(df_display_a, use_container_width=True)

            # Sélecteur de récompense
            st.markdown("#### 🎁 Attribuer une récompense de sauvetage")
            catalog = REWARDS_CATALOG.get(secteur, {})
            rewards_a = catalog.get("sauvetage", ["Aucune récompense configurée pour ce secteur"])
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                selected_reward_a = st.selectbox("Choisir la récompense à offrir", rewards_a, key="reward_a")
            with col_r2:
                st.markdown("<br>", unsafe_allow_html=True)
                if campagnes_actives:
                    if st.button("🚀 Déclencher la campagne", key="btn_a", use_container_width=True):
                        st.success(f"✅ Campagne déclenchée pour {len(df_a_show)} clients — Récompense : **{selected_reward_a}**")
                        st.balloons()
                else:
                    st.warning("Activez les campagnes pour déclencher.")

            # Export
            csv_a = cohorte_a.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exporter Cohorte A (CSV)", csv_a, "cohorte_sauvetage.csv", "text/csv")

    # ── TAB B : FIDÉLITÉ ─────────────────────────────────────────────────────
    with tab_b:
        st.subheader(f"🏆 Cohorte B — Fidélité Historique ({len(cohorte_b)} clients)")
        st.info("Ces clients sont stables et anciens. Les remercier renforce leur attachement à votre marque.")

        if cohorte_b.empty:
            st.info("📊 Aucun client dans la cohorte fidélité pour le moment.")
        else:
            show_cols_b = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnProba', 'Médaille'] if c in cohorte_b.columns]
            df_display_b = cohorte_b[show_cols_b].sort_values('tenure', ascending=False).head(50).copy()
            df_display_b.index = range(1, len(df_display_b) + 1)
            if 'ChurnProba' in df_display_b.columns:
                df_display_b['ChurnProba'] = df_display_b['ChurnProba'].apply(lambda x: f"{x*100:.1f}%")
            df_display_b = df_display_b.rename(columns={
                'tenure': 'Ancienneté (mois)',
                'MonthlyCharges': 'Charges/mois (MAD)',
                'TotalCharges': 'Total cumulé (MAD)',
                'ChurnProba': '✅ Score de risque',
            })
            st.dataframe(df_display_b, use_container_width=True)

            # Récompense fidélité
            st.markdown("#### 🎖️ Attribuer une récompense de fidélité")
            catalog = REWARDS_CATALOG.get(secteur, {})
            rewards_b = catalog.get("fidelite", ["Aucune récompense configurée pour ce secteur"])
            col_r3, col_r4 = st.columns([2, 1])
            with col_r3:
                selected_reward_b = st.selectbox("Choisir la récompense à offrir", rewards_b, key="reward_b")
            with col_r4:
                st.markdown("<br>", unsafe_allow_html=True)
                if campagnes_actives:
                    if st.button("🎖️ Envoyer aux clients fidèles", key="btn_b", use_container_width=True):
                        st.success(f"✅ Récompense envoyée à {len(cohorte_b)} clients fidèles — **{selected_reward_b}**")
                        st.balloons()
                else:
                    st.warning("Activez les campagnes pour envoyer.")

            csv_b = cohorte_b.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exporter Cohorte B (CSV)", csv_b, "cohorte_fidelite.csv", "text/csv")

    # ── TAB CHAMPIONS : MUR DES CHAMPIONS ───────────────────────────────────
    with tab_champ:
        st.subheader(f"🌟 Mur des Champions — Top clients stables ({len(champions)} champions)")
        st.markdown("""
        <div style='background:#0D1B2E;border:1px solid #F59E0B;border-radius:12px;padding:1rem;margin-bottom:1rem;'>
            <p style='color:#F59E0B;font-weight:600;margin:0 0 6px;'>Qui sont les Champions ?</p>
            <p style='color:#CBD5E1;margin:0;font-size:0.9rem;'>
                Clients avec un score de churn < 20% ET une ancienneté > 12 mois.<br>
                Ce sont vos ambassadeurs — ils méritent d'être remerciés chaque mois.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if champions.empty:
            st.info("📊 Aucun champion détecté. Les champions apparaissent quand ChurnProba < 20% ET ancienneté > 12 mois.")
        else:
            top10 = champions.sort_values('tenure', ascending=False).head(10).copy()

            # Cartes Champions
            cols_champ = st.columns(5)
            for i, (_, row) in enumerate(top10.head(10).iterrows()):
                col = cols_champ[i % 5]
                tenure   = int(row.get('tenure', 0))
                score    = row.get('ChurnProba', 0) * 100
                charges  = row.get('MonthlyCharges', 0)
                annees   = tenure // 12
                mois_r   = tenure % 12
                medal    = "🥇" if tenure >= 60 else "🥈" if tenure >= 36 else "🥉"

                col.markdown(f"""
                <div style='background:#0D1B2E;border:1px solid #F59E0B;border-radius:12px;
                            padding:0.9rem;text-align:center;margin-bottom:8px;'>
                    <div style='font-size:1.8rem;'>{medal}</div>
                    <div style='color:#F59E0B;font-weight:700;font-size:0.95rem;'>
                        Client #{i+1}
                    </div>
                    <div style='color:#CBD5E1;font-size:0.8rem;margin-top:4px;'>
                        {annees}a {mois_r}m d'ancienneté
                    </div>
                    <div style='color:#02C39A;font-size:0.8rem;'>
                        Risque : {score:.1f}%
                    </div>
                    <div style='color:#64748B;font-size:0.75rem;'>
                        {charges:.0f} MAD/mois
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Anniversaires ce mois
            st.markdown("---")
            st.subheader("🎂 Anniversaires de contrat ce mois")
            anniversaires = champions[champions['tenure'] % 12 == 0].copy()
            if anniversaires.empty:
                st.info("Aucun anniversaire de contrat ce mois-ci.")
            else:
                anniversaires['Années'] = (anniversaires['tenure'] // 12).astype(str) + " an(s)"
                show_cols_c = [c for c in ['tenure', 'MonthlyCharges', 'ChurnProba', 'Années'] if c in anniversaires.columns]
                df_anniv    = anniversaires[show_cols_c].copy()
                df_anniv.index = range(1, len(df_anniv) + 1)
                if 'ChurnProba' in df_anniv.columns:
                    df_anniv['ChurnProba'] = df_anniv['ChurnProba'].apply(lambda x: f"{x*100:.1f}%")
                df_anniv = df_anniv.rename(columns={'tenure': 'Ancienneté (mois)', 'MonthlyCharges': 'Charges/mois (MAD)', 'ChurnProba': 'Score de risque'})
                st.dataframe(df_anniv, use_container_width=True)
                st.success(f"🎉 {len(anniversaires)} client(s) fêtent leur anniversaire de contrat ce mois — les messages automatiques seront envoyés le 1er du mois à 10h.")

            # Aperçu des messages
            st.markdown("---")
            st.subheader("📧 Aperçu des messages automatiques")
            messages = GRATITUDE_MESSAGES.get(secteur, GRATITUDE_MESSAGES.get("📱 Télécom", {}))

            with st.expander("📅 Message d'anniversaire (exemple)"):
                ex_msg = messages.get("anniversaire", "").format(
                    annees=2, tenure=24,
                    reward="double de vos points fidélité ce mois-ci"
                )
                st.text(ex_msg)

            with st.expander("💌 Message de reconnaissance mensuelle (exemple)"):
                ex_msg2 = messages.get("mensuel", "").format(tenure=18)
                st.text(ex_msg2)

    # ── TAB CONFIG — Panneau d'administration dynamique ─────────────────────
    with tab_config:
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

        # Charger les paramètres existants
        cfg_data = _load_settings(user_email)

        with st.form("loyalty_config_form", border=False):

            # ── BLOC 1 : Seuils ─────────────────────────────────────────────
            st.markdown("### 🎛️ Bloc 1 — Paramétrage des Seuils")
            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                seuil_urgence = st.slider(
                    "🚨 Seuil d'urgence — Cohorte A (%)",
                    min_value=50, max_value=90,
                    value=int(cfg_data["seuil_urgence"] * 100),
                    step=5,
                    help="Les clients dont le score de churn dépasse ce seuil entrent en Cohorte A (Sauvetage).",
                )
            with col_s2:
                tenure_min_b = st.number_input(
                    "🏆 Ancienneté min — Cohorte B (mois)",
                    min_value=1, max_value=120,
                    value=int(cfg_data["tenure_min_b"]),
                    step=1,
                    help="Un client doit avoir au moins N mois d'ancienneté pour intégrer la Cohorte B (Fidélité).",
                )
            with col_s3:
                depense_min = st.number_input(
                    "💰 Dépense minimum requise (MAD)",
                    min_value=0.0, max_value=99999.0,
                    value=float(cfg_data["depense_min"]),
                    step=50.0,
                    help="Filtre de rentabilité : les clients dont la dépense mensuelle est inférieure à ce seuil sont exclus des campagnes.",
                )

            st.markdown("---")

            # ── BLOC 2 : Valeur récompense ───────────────────────────────────
            st.markdown("### 💎 Bloc 2 — Personnalisation de la Valeur")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                reward_type = st.selectbox(
                    "🎁 Type de récompense",
                    ["Pourcentage %", "Montant fixe MAD", "En nature"],
                    index=["Pourcentage %", "Montant fixe MAD", "En nature"].index(
                        cfg_data.get("reward_type", "Pourcentage %")
                    ),
                    help="Choisissez la nature de la récompense offerte aux clients ciblés.",
                )
            with col_v2:
                reward_value = st.number_input(
                    "📊 Valeur de la récompense",
                    min_value=0.0, max_value=99999.0,
                    value=float(cfg_data["reward_value"]),
                    step=5.0,
                    help="Ex : 20 → 20% de remise, ou 200 MAD offerts, ou 1 unité en nature.",
                )

            # Aperçu en temps réel
            _preview_label = {
                "Pourcentage %": f"{reward_value:.0f}% de remise",
                "Montant fixe MAD": f"{reward_value:.0f} MAD offerts",
                "En nature": f"{reward_value:.0f} unité(s) en nature",
            }.get(reward_type, "")
            st.info(f"**Aperçu :** La récompense affichée sera → **{_preview_label}**")

            st.markdown("---")

            # ── BLOC 3 : Moteur de templates ─────────────────────────────────
            st.markdown("### 📝 Bloc 3 — Moteur de Templates")
            st.markdown("""
            <div style='background:#0D1B2E;border:1px solid #2d3748;border-radius:8px;
                        padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#94A3B8;'>
                <strong style='color:#667eea;'>Balises dynamiques disponibles :</strong>
                &nbsp;
                <code style='color:#02C39A;background:#1a1d2e;padding:2px 6px;border-radius:4px;'>{client_nom}</code>
                <code style='color:#02C39A;background:#1a1d2e;padding:2px 6px;border-radius:4px;'>{anciennete_mois}</code>
                <code style='color:#02C39A;background:#1a1d2e;padding:2px 6px;border-radius:4px;'>{valeur_recompense}</code>
                <code style='color:#02C39A;background:#1a1d2e;padding:2px 6px;border-radius:4px;'>{score_risque}</code>
                <code style='color:#02C39A;background:#1a1d2e;padding:2px 6px;border-radius:4px;'>{nom_entreprise}</code>
            </div>
            """, unsafe_allow_html=True)

            email_subject = st.text_input(
                "📧 Objet de l'email / SMS",
                value=cfg_data["email_subject"],
                max_chars=120,
                help="Utilisez les balises ci-dessus pour personnaliser l'objet.",
            )
            email_body = st.text_area(
                "💬 Corps du message",
                value=cfg_data["email_body"],
                height=200,
                help="Corps complet du message. Les balises {…} seront remplacées automatiquement.",
            )

            # Aperçu du message
            with st.expander("👁️ Aperçu du message (exemple avec données fictives)"):
                _demo = {
                    "client_nom":       "Mohamed Alami",
                    "anciennete_mois":  "24",
                    "valeur_recompense": _preview_label,
                    "score_risque":     "72%",
                    "nom_entreprise":   user_company or "RetainIQ",
                }
                try:
                    _subj_preview = email_subject.format(**_demo)
                    _body_preview = email_body.format(**_demo)
                    st.markdown(f"**Objet :** {_subj_preview}")
                    st.text(_body_preview)
                except KeyError as e:
                    st.warning(f"Balise inconnue dans le template : {e}")

            st.markdown("---")

            # ── BLOC 4 : Garde-fous ──────────────────────────────────────────
            st.markdown("### 🔒 Bloc 4 — Garde-Fous et Limites (Sécurité)")
            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                budget_max = st.number_input(
                    "💵 Budget max / mois (MAD)",
                    min_value=0.0, max_value=9_999_999.0,
                    value=float(cfg_data["budget_max"]),
                    step=500.0,
                    help="Montant total maximum alloué aux récompenses sur un mois calendaire. 0 = illimité.",
                )
            with col_g2:
                quota_mois = st.number_input(
                    "👥 Quota de récompenses / mois",
                    min_value=0, max_value=100_000,
                    value=int(cfg_data["quota_mois"]),
                    step=10,
                    help="Nombre maximum de clients pouvant recevoir une récompense par mois. 0 = illimité.",
                )
            with col_g3:
                periode_carence = st.number_input(
                    "⏳ Période de carence (mois)",
                    min_value=0, max_value=24,
                    value=int(cfg_data["periode_carence"]),
                    step=1,
                    help="Un même client ne peut pas recevoir une récompense plus d'une fois par N mois.",
                )

            st.markdown("---")

            # ── BLOC 5 : Activation globale ──────────────────────────────────
            st.markdown("### 🔌 Bloc 5 — Activation Globale des Campagnes")
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                campagne_sauvetage = st.toggle(
                    "🚨 Activer la campagne **Sauvetage** (Cohorte A)",
                    value=bool(cfg_data["campagne_sauvetage"]),
                    help="Active ou désactive les actions automatiques vers les clients à fort risque de churn.",
                )
                if campagne_sauvetage:
                    st.success("✅ Campagne Sauvetage **active**")
                else:
                    st.warning("💤 Campagne Sauvetage **désactivée**")

            with col_t2:
                campagne_fidelite = st.toggle(
                    "🏆 Activer la campagne **Fidélité** (Cohorte B)",
                    value=bool(cfg_data["campagne_fidelite"]),
                    help="Active ou désactive les récompenses vers les clients stables et anciens.",
                )
                if campagne_fidelite:
                    st.success("✅ Campagne Fidélité **active**")
                else:
                    st.warning("💤 Campagne Fidélité **désactivée**")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── BOUTON SAUVEGARDER ───────────────────────────────────────────
            col_save, col_reset, _ = st.columns([1, 1, 2])
            submitted = col_save.form_submit_button(
                "💾 Sauvegarder la configuration",
                use_container_width=True,
                type="primary",
            )
            reset = col_reset.form_submit_button(
                "↩️ Réinitialiser",
                use_container_width=True,
            )

        if submitted:
            new_settings = {
                "seuil_urgence":      seuil_urgence / 100.0,
                "tenure_min_b":       int(tenure_min_b),
                "depense_min":        float(depense_min),
                "reward_type":        reward_type,
                "reward_value":       float(reward_value),
                "email_subject":      email_subject,
                "email_body":         email_body,
                "budget_max":         float(budget_max),
                "quota_mois":         int(quota_mois),
                "periode_carence":    int(periode_carence),
                "campagne_sauvetage": campagne_sauvetage,
                "campagne_fidelite":  campagne_fidelite,
            }
            _save_settings(user_email, new_settings)
            st.success("✅ Configuration sauvegardée avec succès !")
            st.balloons()

        if reset:
            _save_settings(user_email, _DEFAULT_SETTINGS.copy())
            st.info("↩️ Configuration réinitialisée aux valeurs par défaut.")
            st.rerun()