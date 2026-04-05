import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CALCUL DES VALEURS SHAP
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_shap_values(_model, _X):
    """
    Calcule les valeurs SHAP pour le modèle et les données.
    Mis en cache pour ne pas recalculer à chaque interaction.
    """
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X)
    return shap_values, explainer.expected_value

def get_shap_explanation_text(shap_vals, feature_names, top_n=3):
    """
    Génère une explication en langage naturel à partir des valeurs SHAP.
    Ex: "Ce client risque de partir principalement à cause de son contrat mensuel..."
    """
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap":    shap_vals
    }).sort_values("shap", ascending=False)

    top_push    = shap_df[shap_df["shap"] > 0].head(top_n)
    top_protect = shap_df[shap_df["shap"] < 0].tail(top_n)

    reasons_push    = []
    reasons_protect = []

    # Mapping des noms techniques vers langage naturel
    LABEL_MAP = {
        "tenure":                        "la faible ancienneté",
        "MonthlyCharges":                "les charges mensuelles élevées",
        "TotalCharges":                  "le total des charges",
        "SeniorCitizen":                 "le statut senior",
        "Contract_Month-to-month":       "le contrat mensuel",
        "Contract_One year":             "le contrat d'un an",
        "Contract_Two year":             "le contrat de deux ans",
        "Contract_One_year":             "le contrat d'un an",
        "Contract_Two_year":             "le contrat de deux ans",
        "InternetService_Fiber optic":   "la fibre optique sans sécurité",
        "InternetService_Fiber_optic":   "la fibre optique sans sécurité",
        "InternetService_No":            "l'absence de service internet",
        "OnlineSecurity_Yes":            "la sécurité en ligne",
        "TechSupport_Yes":               "le support technique",
        "PaperlessBilling_Yes":          "la facturation dématérialisée",
        "PaymentMethod_Electronic check":"le paiement par chèque électronique",
        "PaymentMethod_Electronic_check":"le paiement par chèque électronique",
        "Partner_Yes":                   "la présence d'un partenaire",
        "Dependents_Yes":                "la présence de dépendants",
        "gender_Male":                   "le genre masculin",
    }

    for _, row in top_push.iterrows():
        feat  = row["feature"]
        label = LABEL_MAP.get(feat, feat.replace("_", " "))
        reasons_push.append(f"**{label}** (+{row['shap']:.3f})")

    for _, row in top_protect.iterrows():
        feat  = row["feature"]
        label = LABEL_MAP.get(feat, feat.replace("_", " "))
        reasons_protect.append(f"**{label}** ({row['shap']:.3f})")

    text = ""
    if reasons_push:
        text += "**Facteurs qui augmentent le risque :** " + ", ".join(reasons_push) + "\n\n"
    if reasons_protect:
        text += "**Facteurs qui réduisent le risque :** " + ", ".join(reasons_protect)

    return text

# ═══════════════════════════════════════════════════════════════
# PAGE SHAP COMPLÈTE
# ═══════════════════════════════════════════════════════════════
def show_shap_page(model, df, feature_names):
    """
    Page complète d'Explainable AI avec SHAP.
    Affiche 3 types de visualisations + explication par client.
    """

    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🧠 Explainable AI — SHAP</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>
            Comprendre POURQUOI le modèle prédit le départ de chaque client
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **SHAP** (SHapley Additive exPlanations) explique chaque prédiction individuellement. "
            "Au lieu d'une boîte noire, vous voyez exactement quelles variables poussent un client à partir.")

    # Préparer les données
    drop_cols = [c for c in ["Churn", "ChurnProba", "RiskLevel"] if c in df.columns]
    X         = df.drop(columns=drop_cols)
    X         = X[list(feature_names)]

    # Calculer SHAP
    with st.spinner("Calcul des valeurs SHAP en cours..."):
        shap_values, expected_value = compute_shap_values(model, X)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # ── VUE 1 : IMPORTANCE GLOBALE ──────────────────────────────
    st.markdown("---")
    st.subheader("📊 Vue 1 — Importance globale des variables")
    st.caption("Quelles variables ont le plus d'impact sur les prédictions, en moyenne sur tous les clients ?")

    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=True)
    top_features  = mean_abs_shap.tail(15)

    # Noms lisibles
    LABEL_MAP = {
        "tenure":                        "Ancienneté (mois)",
        "MonthlyCharges":                "Charges mensuelles",
        "TotalCharges":                  "Total charges",
        "SeniorCitizen":                 "Client senior",
        "Contract_Month-to-month":       "Contrat mensuel",
        "Contract_One year":             "Contrat 1 an",
        "Contract_Two year":             "Contrat 2 ans",
        "Contract_One_year":             "Contrat 1 an",
        "Contract_Two_year":             "Contrat 2 ans",
        "InternetService_Fiber optic":   "Fibre optique",
        "InternetService_Fiber_optic":   "Fibre optique",
        "InternetService_No":            "Pas d'internet",
        "OnlineSecurity_Yes":            "Sécurité en ligne",
        "TechSupport_Yes":               "Support technique",
        "PaperlessBilling_Yes":          "Facturation dématérialisée",
        "PaymentMethod_Electronic check":"Chèque électronique",
        "PaymentMethod_Electronic_check":"Chèque électronique",
        "Partner_Yes":                   "A un partenaire",
        "Dependents_Yes":                "A des dépendants",
        "gender_Male":                   "Genre masculin",
    }

    labels = [LABEL_MAP.get(f, f.replace("_", " ")) for f in top_features.index]

    fig = go.Figure(go.Bar(
        x=top_features.values,
        y=labels,
        orientation="h",
        marker=dict(
            color=top_features.values,
            colorscale=[[0, "#1E3A5F"], [0.5, "#0E6BA8"], [1, "#02C39A"]],
            showscale=False
        )
    ))
    fig.update_layout(
        height=420,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Importance SHAP moyenne (|valeur|)",
            color="#CBD5E1",
            gridcolor="#1E3A5F"
        ),
        yaxis=dict(color="#CBD5E1"),
        font=dict(color="#CBD5E1", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── VUE 2 : IMPACT POSITIF VS NÉGATIF ───────────────────────
    st.markdown("---")
    st.subheader("📈 Vue 2 — Impact positif vs négatif sur le churn")
    st.caption("En rouge : variables qui augmentent le risque de churn. En vert : variables qui le réduisent.")

    mean_shap = shap_df.mean().sort_values(ascending=True)
    top_signed = pd.concat([mean_shap.head(8), mean_shap.tail(8)]).drop_duplicates()
    top_signed = top_signed.sort_values()

    colors = ["#EF4444" if v > 0 else "#02C39A" for v in top_signed.values]
    labels_signed = [LABEL_MAP.get(f, f.replace("_", " ")) for f in top_signed.index]

    fig2 = go.Figure(go.Bar(
        x=top_signed.values,
        y=labels_signed,
        orientation="h",
        marker_color=colors
    ))
    fig2.add_vline(x=0, line_color="#64748B", line_width=1)
    fig2.update_layout(
        height=420,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Impact moyen sur le risque de churn",
            color="#CBD5E1",
            gridcolor="#1E3A5F"
        ),
        yaxis=dict(color="#CBD5E1"),
        font=dict(color="#CBD5E1", size=12)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Interprétation textuelle
    top3_push    = mean_shap.tail(3).index[::-1].tolist()
    top3_protect = mean_shap.head(3).index.tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#1C0A0A;border-left:4px solid #EF4444;
                    border-radius:8px;padding:1rem;'>
            <p style='color:#FCA5A5;font-weight:600;margin:0 0 8px;'>
                🔴 Facteurs qui augmentent le churn
            </p>
        """ + "".join([
            f"<p style='color:#CBD5E1;margin:4px 0;font-size:0.9rem;'>› {LABEL_MAP.get(f, f.replace('_',' '))}</p>"
            for f in top3_push
        ]) + "</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:#0A1C0F;border-left:4px solid #02C39A;
                    border-radius:8px;padding:1rem;'>
            <p style='color:#6EE7B7;font-weight:600;margin:0 0 8px;'>
                🟢 Facteurs qui réduisent le churn
            </p>
        """ + "".join([
            f"<p style='color:#CBD5E1;margin:4px 0;font-size:0.9rem;'>› {LABEL_MAP.get(f, f.replace('_',' '))}</p>"
            for f in top3_protect
        ]) + "</div>", unsafe_allow_html=True)

    # ── VUE 3 : EXPLICATION CLIENT PAR CLIENT ───────────────────
    st.markdown("---")
    st.subheader("🔍 Vue 3 — Explication individuelle par client")
    st.caption("Sélectionnez un client pour voir exactement pourquoi le modèle prédit son départ.")

    # Sélecteur de client
    n_clients     = min(len(df), 100)
    client_labels = [f"Client #{i+1}" for i in range(n_clients)]

    col1, col2 = st.columns([2, 1])
    with col1:
        client_idx = st.select_slider(
            "Choisir un client",
            options=list(range(n_clients)),
            format_func=lambda x: f"Client #{x+1}",
            value=0
        )
    with col2:
        # Afficher le score de risque du client sélectionné
        if "ChurnProba" in df.columns:
            score   = df.iloc[client_idx]["ChurnProba"]
            color   = "#EF4444" if score > 0.6 else "#F59E0B" if score > 0.35 else "#02C39A"
            label   = "Risque Élevé" if score > 0.6 else "Risque Modéré" if score > 0.35 else "Risque Faible"
            st.markdown(f"""
            <div style='background:#0D1B2E;border:2px solid {color};border-radius:12px;
                        padding:1rem;text-align:center;margin-top:10px;'>
                <div style='font-size:1.8rem;font-weight:800;color:{color};'>{score*100:.1f}%</div>
                <div style='color:{color};font-size:0.85rem;font-weight:600;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Valeurs SHAP du client sélectionné
    client_shap = shap_values[client_idx]
    client_shap_df = pd.DataFrame({
        "feature": list(feature_names),
        "shap":    client_shap,
        "label":   [LABEL_MAP.get(f, f.replace("_", " ")) for f in feature_names],
        "value":   X.iloc[client_idx].values
    }).sort_values("shap", key=abs, ascending=True).tail(12)

    # Graphique waterfall simplifié
    colors_client = ["#EF4444" if v > 0 else "#02C39A" for v in client_shap_df["shap"]]

    fig3 = go.Figure(go.Bar(
        x=client_shap_df["shap"],
        y=client_shap_df["label"],
        orientation="h",
        marker_color=colors_client,
        text=[f"{v:+.3f}" for v in client_shap_df["shap"]],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=11)
    ))
    fig3.add_vline(x=0, line_color="#64748B", line_width=1.5)
    fig3.update_layout(
        title=dict(text=f"Facteurs SHAP — Client #{client_idx+1}", font=dict(color="#F1F5F9")),
        height=420,
        margin=dict(t=40, b=20, l=20, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Impact sur la probabilité de churn",
            color="#CBD5E1",
            gridcolor="#1E3A5F"
        ),
        yaxis=dict(color="#CBD5E1"),
        font=dict(color="#CBD5E1", size=12)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Explication en langage naturel
    explanation = get_shap_explanation_text(client_shap, feature_names)
    st.markdown(f"""
    <div style='background:#0D1B2E;border:1px solid #1E3A5F;border-radius:12px;padding:1.2rem;'>
        <p style='color:#02C39A;font-weight:600;margin:0 0 10px;font-size:0.95rem;'>
            💬 Explication en langage naturel — Client #{client_idx+1}
        </p>
        <p style='color:#CBD5E1;margin:0;font-size:0.9rem;line-height:1.7;'>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    # Profil complet du client
    with st.expander(f"📋 Profil complet du Client #{client_idx+1}"):
        client_data = df.iloc[client_idx]
        show_fields = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        show_fields = [f for f in show_fields if f in client_data.index]

        cols = st.columns(len(show_fields))
        labels_profile = {
            "tenure":         "Ancienneté (mois)",
            "MonthlyCharges": "Charges mensuelles",
            "TotalCharges":   "Total charges",
            "SeniorCitizen":  "Client senior",
        }
        for i, field in enumerate(show_fields):
            val = client_data[field]
            if isinstance(val, float):
                val = f"{val:.2f}"
            cols[i].metric(labels_profile.get(field, field), val)

    # ── VUE 4 : SCATTER — CHARGES VS SHAP ───────────────────────
    st.markdown("---")
    st.subheader("📉 Vue 4 — Relation entre charges et impact SHAP")
    st.caption("Chaque point est un client. Plus il est en haut à droite, plus ses charges contribuent à son churn.")

    if "MonthlyCharges" in feature_names:
        mc_idx      = list(feature_names).index("MonthlyCharges")
        mc_shap     = shap_values[:, mc_idx]
        mc_values   = X["MonthlyCharges"].values

        scatter_df = pd.DataFrame({
            "Charges mensuelles (€)": mc_values,
            "Impact SHAP":            mc_shap,
            "Score de risque (%)":    df["ChurnProba"].values * 100 if "ChurnProba" in df.columns else np.zeros(len(df))
        })

        fig4 = px.scatter(
            scatter_df,
            x="Charges mensuelles (€)",
            y="Impact SHAP",
            color="Score de risque (%)",
            color_continuous_scale=[[0, "#02C39A"], [0.5, "#F59E0B"], [1, "#EF4444"]],
            title="Impact des charges mensuelles sur le churn (SHAP)",
            opacity=0.7
        )
        fig4.add_hline(y=0, line_color="#64748B", line_dash="dash", line_width=1)
        fig4.update_layout(
            height=380,
            margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=12),
            xaxis=dict(color="#CBD5E1", gridcolor="#1E3A5F"),
            yaxis=dict(color="#CBD5E1", gridcolor="#1E3A5F"),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Points au-dessus de 0 = les charges augmentent le risque de churn. En dessous = elles le réduisent.")