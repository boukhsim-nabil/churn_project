import datetime
import io
import os
import tempfile
import warnings

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_pipeline import (
    show_pipeline_page, load_user_model, load_tenant_model,
    triage_risque, _sanitize_company, TENANT_DATA_DIR,
    prepare_features_for_prediction,
)
from shap_explainer import show_shap_page
from email_reports import generate_pdf_report, send_pdf_via_sendgrid
from email_service import send_campaign_email
from auth import show_auth_page
from loyalty_page import show_loyalty_page
import scheduler as sched

warnings.filterwarnings('ignore')

# ── Mapping catégoriel : selectboxes propres → colonnes dummifiées ──────────
# Pour chaque variable catégorielle du dataset métier, on définit :
#   label       : libellé affiché dans l'UI
#   options     : valeurs humaines proposées dans le selectbox
#   dummy_cols  : colonnes dummifiées réellement attendues par le modèle
#   encoding    : pour chaque option, la liste de 0/1 à écrire dans dummy_cols
CATEGORICAL_DUMMIES_MAP = {
    "type_engagement": {
        "label":      "Type d'engagement",
        "options":    ["Engagement annuel", "Sans engagement"],
        "dummy_cols": ["type_engagement_Sans engagement"],
        "encoding": {
            "Engagement annuel": [0],
            "Sans engagement":   [1],
        },
    },
    "pack_service": {
        "label":      "Pack Service",
        "options":    ["Starter", "Premium"],
        "dummy_cols": ["pack_service_Starter"],
        "encoding": {
            "Starter": [1],
            "Premium": [0],
        },
    },
}

def _get_active_cat_groups(feature_names: list) -> dict:
    """Retourne les groupes catégoriels dont au moins une dummy col est présente dans feature_names."""
    fn_set = set(feature_names)
    return {
        key: info
        for key, info in CATEGORICAL_DUMMIES_MAP.items()
        if any(col in fn_set for col in info["dummy_cols"])
    }

# ── Vérification de la session ──────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_auth_page()
    st.stop()  # ← IMPORTANT : arrête tout le reste si pas connecté

# ── L'utilisateur est connecté — on récupère ses infos ──────────
user_company = st.session_state.get("user_company", "Mon Entreprise")
user_secteur = st.session_state.get("user_secteur", "📱 Télécom")
user_role    = st.session_state.get("user_role",    "agent")

# Helpers RBAC  (hiérarchie : super_admin > admin > manager > agent)
_is_super_admin      = user_role == "super_admin"
_is_admin            = user_role in ("admin", "super_admin")
_is_manager_or_admin = user_role in ("manager", "admin", "super_admin")

_ROLE_LABELS = {
    "super_admin": ("🔮 Super Admin", "#8B5CF6"),
    "admin":       ("⚙️ Admin",       "#EF4444"),
    "manager":     ("📊 Manager",     "#F59E0B"),
    "agent":       ("👤 Agent",       "#10B981"),
}

# ── Démarrage du scheduler (une seule fois par processus) ────────
if not st.session_state.get("_scheduler_started"):
    sched.start_scheduler()
    st.session_state["_scheduler_started"] = True

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔮 RetainIQ — Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "RetainIQ — Plateforme IA de Prédiction du Churn — Projet Industriel 2024-2025"}
)

# ══════════════════════════════════════════════════════════════════
# CSS — original conservé + nouvelles classes ajoutées
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── CSS ORIGINAL ─────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stSelectbox > div > div {
        background-color: #3a3c4a;
        border-radius: 10px;
    }
    h1 { color: #2c3e50; font-weight: 700; }
    h2, h3 { color: #34495e; font-weight: 600; }

    /* ── NOUVELLES CLASSES ────────────────────────────────────── */
    .risk-high   { background:#1C0A0A; border:2px solid #EF4444; border-radius:16px; padding:1.5rem; text-align:center; margin:0.5rem 0; }
    .risk-medium { background:#1C150A; border:2px solid #F59E0B; border-radius:16px; padding:1.5rem; text-align:center; margin:0.5rem 0; }
    .risk-low    { background:#0A1C0F; border:2px solid #00CC96; border-radius:16px; padding:1.5rem; text-align:center; margin:0.5rem 0; }
    .section-card {
        background: #1a1d2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        background: #2D1515;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #FCA5A5;
    }
    .chat-user { background:linear-gradient(135deg,#667eea,#764ba2); border-radius:16px 16px 2px 16px; padding:0.8rem 1.2rem; margin:0.5rem 0; color:white; max-width:80%; margin-left:auto; }
    .chat-bot  { background:#1a1d2e; border:1px solid #2d3748; border-radius:16px 16px 16px 2px; padding:0.8rem 1.2rem; margin:0.5rem 0; color:#CBD5E1; max-width:85%; }
    .secteur-badge { background:linear-gradient(135deg,#667eea,#764ba2); border-radius:8px; padding:8px 14px; color:white; font-size:0.85rem; font-weight:600; text-align:center; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
           padding:20px;border-radius:15px;margin-bottom:20px;text-align:center;'>
    <h2 style='color:white;margin:0;'>🔮 RetainIQ</h2>
    <p style='color:#e8f4f8;margin:5px 0 0 0;'>Prédiction Intelligente du Churn</p>
</div>
""", unsafe_allow_html=True)

# Filtre secteur (NOUVEAU)
secteur = st.session_state.get("user_secteur", "📱 Télécom")

st.sidebar.markdown(f"""
<div style='background:#1a1d2e;border:1px solid #2d3748;
            border-radius:8px;padding:12px;margin-bottom:8px;'>
    <p style='color:#888;font-size:0.75rem;margin:0;'>🏭 Secteur d'activité</p>
    <p style='color:#667eea;font-weight:600;margin:6px 0 0 0;
              font-size:1rem;'>{secteur}</p>
    <p style='color:#555;font-size:0.72rem;margin:4px 0 0 0;'>
        Défini à l'inscription · Non modifiable
    </p>
</div>
""", unsafe_allow_html=True)
SECTEUR_CONFIG = {
    "📱 Télécom":        {"tenure": "Ancienneté (mois)",  "charges": "Forfait mensuel (€)",  "churn_label": "Résiliation"},
    "💪 Salle de Sport": {"tenure": "Mois d'abonnement",  "charges": "Abonnement (€/mois)",  "churn_label": "Non-renouvellement"},
    "🛍️ E-commerce":     {"tenure": "Mois client",        "charges": "Panier moyen (€)",     "churn_label": "Inactivité"},
    "🎓 EdTech":         {"tenure": "Mois inscrit",       "charges": "Abonnement (€/mois)",  "churn_label": "Désinscription"},
    "☁️ SaaS B2B":       {"tenure": "Mois client",        "charges": "MRR (€)",              "churn_label": "Résiliation"},
}
_DEFAULT_CFG = {"tenure": "Mois client", "charges": "Montant mensuel (€)", "churn_label": "Résiliation"}
cfg = SECTEUR_CONFIG.get(secteur, _DEFAULT_CFG)

st.sidebar.markdown("---")

# ── VÉRIFICATION DU MODÈLE (Blank Slate) ──
import os
user_email = st.session_state.get("user_email", "")
_email_safe = user_email.replace("@", "_at_").replace(".", "_")
has_model = os.path.exists(f"model_{_email_safe}.pkl")
# Agents sans modèle personnel : déverrouiller la navigation si le modèle
# partagé de l'entreprise existe déjà (chargé plus bas par load_tenant_model).
if not has_model and user_company:
    _company_safe = _sanitize_company(user_company)
    has_model = os.path.exists(os.path.join(TENANT_DATA_DIR, f"{_company_safe}_model.pkl"))

# ── MENU DE NAVIGATION RBAC ─────────────────────────────────────
# Pages accessibles à tous les rôles
_pages_all = [
    "🏠 Overview",
    "🔮 AI Prediction",
    "⚡ Simulateur What-If",
    "🚨 Alertes Clients",
    "🤖 Assistant IA",
    "🏆 Programme de Fidélité",
]
# Pages réservées Manager + Admin
_pages_manager = [
    "📊 Visual Analytics",
    "🌟 Future Scenarios",
    "📤 Importer mes données",
    "🧠 Explainable AI",
    "📧 Campagnes & Rapports",
]
# Page réservée Manager + Admin
_pages_admin = ["⚙️ Panneau Admin"]

if not has_model:
    # Blank Slate : tous les utilisateurs peuvent importer leurs données
    _blank_pages = ["🏠 Bienvenue", "📤 Importer mes données"]
    if _is_manager_or_admin:
        _blank_pages.append("⚙️ Panneau Admin")

    st.sidebar.markdown("""
    <div style='background:#1C150A;border:1px solid #F59E0B;border-radius:8px;
                padding:10px 12px;margin-bottom:8px;'>
        <p style='color:#F59E0B;font-size:0.8rem;margin:0;font-weight:600;'>
            ⚠️ Importez vos données pour débloquer toutes les pages
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Redirection programmatique : on écrit directement dans la clé du widget
    # car Streamlit ignore index= si le widget a déjà un état en session_state.
    nav_demandee = st.session_state.pop("_nav_override", None)
    if nav_demandee and nav_demandee in _blank_pages:
        st.session_state["_blank_nav_radio"] = nav_demandee

    section = st.sidebar.radio("🎯 Navigation", _blank_pages, key="_blank_nav_radio")
else:
    # Menu complet filtré par rôle
    _full_pages = list(_pages_all)
    if _is_manager_or_admin:
        _full_pages += _pages_manager
        _full_pages += _pages_admin

    section = st.sidebar.radio("🎯 Navigation", _full_pages)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='secteur-badge'>Secteur : {secteur}</div>", unsafe_allow_html=True)
_role_label, _role_color = _ROLE_LABELS.get(user_role, ("👤 Agent", "#10B981"))
st.sidebar.markdown(f"""
<div style='background:#1a1d2e;border:1px solid #2d3748;
            border-radius:8px;padding:12px;margin-bottom:10px;'>
    <p style='color:#888;font-size:0.75rem;margin:0;'>Connecté en tant que</p>
    <p style='color:white;font-weight:600;margin:4px 0 0 0;
              font-size:0.9rem;'>{user_company}</p>
    <p style='color:#667eea;font-size:0.8rem;margin:2px 0 0 0;'>{user_secteur}</p>
    <span style='display:inline-block;margin-top:6px;padding:3px 10px;
                 border-radius:12px;background:{_role_color}22;
                 border:1px solid {_role_color};color:{_role_color};
                 font-size:0.75rem;font-weight:700;'>{_role_label}</span>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Se déconnecter", use_container_width=True):
    st.session_state.logged_in  = False
    st.session_state.user_email = ""
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# DONNÉES — identique à l'original
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Telco-Customer-Churn.csv")
        df.drop('customerID', axis=1, inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df = pd.get_dummies(df, drop_first=True)
        return df, True
    except:
        np.random.seed(42)
        n_samples = 1000
        data = {
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples).clip(20, 120),
            'TotalCharges': np.random.normal(2500, 1500, n_samples).clip(100, 8000),
            'gender_Male': np.random.choice([0, 1], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner_Yes': np.random.choice([0, 1], n_samples),
            'Dependents_Yes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'PhoneService_Yes': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'InternetService_Fiber_optic': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'InternetService_No': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'OnlineSecurity_Yes': np.random.choice([0, 1], n_samples),
            'Contract_One_year': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Contract_Two_year': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'PaperlessBilling_Yes': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'PaymentMethod_Electronic_check': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        df = pd.DataFrame(data)
        churn_prob = (
            0.1 +
            (df['tenure'] < 12) * 0.3 +
            (df['MonthlyCharges'] > 80) * 0.2 +
            df['SeniorCitizen'] * 0.15 +
            (df['Contract_One_year'] == 0) * (df['Contract_Two_year'] == 0) * 0.25 +
            df['PaymentMethod_Electronic_check'] * 0.1
        ).clip(0, 0.8)
        df['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        return df, False

df, is_real_data = load_data()

# ══════════════════════════════════════════════════════════════════
# MODÈLE — identique à l'original
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, X.columns, accuracy

model, feature_names, model_accuracy = train_model(df)

# Charger le modèle custom si disponible
custom_model, custom_features, custom_df = load_user_model(user_email)
_tenant_fallback = False

if custom_model is None:
    # Fallback : modèle partagé de l'entreprise (chargé par un admin/manager)
    custom_model, custom_features, custom_df, _tenant_metrics = load_tenant_model(user_company)
    _tenant_fallback = custom_model is not None
    if _tenant_fallback:
        st.session_state["custom_model_trained"] = True
        if _tenant_metrics is not None:
            st.session_state["custom_model_metrics"] = _tenant_metrics

if custom_model is not None and custom_features is not None and custom_df is not None:
    model = custom_model
    feature_names = custom_features
    df = custom_df.copy()
    if 'ChurnProba' not in df.columns:
        # Alignement garanti : sélectionne exactement les features d'entraînement dans le bon ordre
        X_all_custom = prepare_features_for_prediction(df, custom_features)
        df['ChurnProba'] = model.predict_proba(X_all_custom)[:, 1]
    if 'RiskLevel' not in df.columns:
        df['RiskLevel'] = df['ChurnProba'].apply(
            lambda x: "🔴 Risque Élevé" if x > 0.6 else ("🟡 Risque Modéré" if x > 0.35 else "🟢 Risque Faible")
        )
    if _tenant_fallback:
        st.sidebar.success("✅ Modèle entreprise actif")
    else:
        st.sidebar.success("✅ Modèle personnalisé actif")
else:
    if not has_model:
        st.sidebar.warning("⚠️ Aucun modèle importé")
    else:
        st.sidebar.info("📊 Données démo (Telco)")
    X_all = df.drop("Churn", axis=1)
    df['ChurnProba'] = model.predict_proba(X_all)[:, 1]
    df['RiskLevel']  = df['ChurnProba'].apply(
        lambda x: "🔴 Risque Élevé" if x > 0.6 else ("🟡 Risque Modéré" if x > 0.35 else "🟢 Risque Faible")
    )

# ══════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES (NOUVELLES)
# ══════════════════════════════════════════════════════════════════
def risk_gauge(score):
    if score > 0.6:
        color, label, css_class = "#EF4444", "RISQUE ÉLEVÉ", "risk-high"
    elif score > 0.35:
        color, label, css_class = "#F59E0B", "RISQUE MODÉRÉ", "risk-medium"
    else:
        color, label, css_class = "#00CC96", "RISQUE FAIBLE", "risk-low"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': "%", 'font': {'size': 38, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "#1a1d2e",
            'bordercolor': "#2d3748",
            'steps': [
                {'range': [0, 35],   'color': '#0A1C0F'},
                {'range': [35, 60],  'color': '#1C150A'},
                {'range': [60, 100], 'color': '#1C0A0A'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'value': score * 100}
        }
    ))
    fig.update_layout(height=220, margin=dict(t=20, b=10, l=20, r=20),
                      paper_bgcolor='rgba(0,0,0,0)', font={'color': '#CBD5E1'})
    return fig, label, color, css_class

def get_recommendations(score, tenure, charges):
    recs = []
    if score > 0.6:
        recs.append("🎁 Offrir une remise de 15-20%")
        recs.append("📞 Appel de rétention dans les 48h")
        if tenure < 12:
            recs.append("📋 Proposer un contrat annuel")
        if charges > 80:
            recs.append("💰 Étudier un forfait moins cher")
        if len(recs) < 3:
            recs.append("🔧 Vérifier les incidents en attente")
    elif score > 0.35:
        recs.append("📧 Enquête de satisfaction")
        recs.append("🎯 Proposer des services complémentaires")
        recs.append("📊 Surveiller les 30 prochains jours")
    else:
        recs.append("✅ Client stable — continuer la qualité")
        recs.append("📈 Opportunité d'upsell premium")
        recs.append("⭐ Inviter au programme fidélité")
    return recs[:3]

def build_input_df(tenure, charges, contract, internet, security):
    inp = {}
    for feat in feature_names:
        inp[feat] = float(df[feat].median()) if feat in df.columns and df[feat].nunique() > 2 else 0
    inp['tenure']          = tenure
    inp['MonthlyCharges']  = charges
    inp['TotalCharges']    = tenure * charges
    for col in feature_names:
        if 'One' in col and 'Contract' in col:   inp[col] = 1 if contract == "1 an" else 0
        elif 'Two' in col and 'Contract' in col:  inp[col] = 1 if contract == "2 ans" else 0
        elif 'Fiber' in col:                      inp[col] = 1 if internet == "Fibre optique" else 0
        elif 'No' in col and 'Internet' in col:   inp[col] = 1 if internet == "Aucun" else 0
        elif 'OnlineSecurity' in col:             inp[col] = int(security)
    return pd.DataFrame([inp])[feature_names]

# triage_risque est importé depuis data_pipeline


def gemini_draft_email(context: str, email_type: str) -> str:
    """Generate a professional email draft using Gemini based on context and type."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Clé GEMINI_API_KEY manquante dans le fichier `.env`."

    _secteur  = st.session_state.get("user_secteur", "Non défini")
    _company  = st.session_state.get("user_company", "Notre Entreprise")

    prompt = (
        f"Tu es un expert en marketing de rétention client B2B pour le secteur {_secteur}. "
        f"Rédige un email professionnel, chaleureux et persuasif en français. "
        f"Type d'email : {email_type}. Contexte : {context}. "
        f"L'email est envoyé au nom de '{_company}'. "
        f"Format : commence par 'Objet : ...' sur la première ligne, "
        f"puis une ligne vide, puis le corps de l'email. "
        f"Texte brut uniquement, pas de HTML. Sois concis (max 200 mots)."
    )

    genai.configure(api_key=api_key)
    try:
        _model_ai = genai.GenerativeModel("gemini-2.5-flash")
        response  = _model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Erreur Gemini : {e}"


def gemini_chat_response(question: str, df_clean: pd.DataFrame) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Clé GEMINI_API_KEY manquante dans le fichier `.env`."

    n_total   = len(df_clean)
    n_urgent  = int((df_clean["ChurnProba"] > 0.6).sum()) if "ChurnProba" in df_clean.columns else 0
    secteur   = st.session_state.get("user_secteur", "Non défini")

    system_prompt = (
        "Tu es un expert en Data Science et fidélisation client B2B, intégré au logiciel RetainIQ. "
        "Ton rôle est d'aider les équipes métier à comprendre et réduire le churn client. "
        "Réponds toujours en français, de façon concise, professionnelle et actionnable. "
        "Utilise du Markdown (gras, listes) pour structurer tes réponses. "
        "Ne réponds jamais à des sujets sans rapport avec la fidélisation ou le churn.\n\n"
        f"Contexte actuel du tableau de bord:\n"
        f"- Secteur d'activité: {secteur}\n"
        f"- Nombre total de clients analysés: {n_total:,}\n"
        f"- Clients en risque élevé (ChurnProba > 60%): {n_urgent}\n"
    )

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        full_prompt = system_prompt + "\n\nQuestion de l'utilisateur : " + question
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Erreur Gemini : {e}"

# ══════════════════════════════════════════════════════════════════
# PAGE 0 — BIENVENUE / BLANK SLATE (nouvel utilisateur sans modèle)
# ══════════════════════════════════════════════════════════════════
if section == "🏠 Bienvenue":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0D1B2E 0%,#1a1d2e 100%);
                border:1px solid #667eea;border-radius:20px;
                padding:3rem 2.5rem;text-align:center;margin-bottom:2rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>🔮</div>
        <h1 style='color:white;margin:0 0 0.5rem 0;font-size:2.4rem;'>
            Bienvenue sur RetainIQ
        </h1>
        <p style='color:#94A3B8;font-size:1.1rem;margin:0;'>
            Votre plateforme IA de prédiction du churn — prête à être configurée
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📤", "Importez vos données", "Glissez votre fichier CSV — la détection des colonnes est automatique."),
        (c2, "🤖", "L'IA s'entraîne", "Un modèle XGBoost personnalisé est créé en quelques secondes."),
        (c3, "📊", "Explorez vos insights", "Tableaux de bord, alertes, SHAP, rapports PDF — tout se déverrouille."),
    ]:
        col.markdown(f"""
        <div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:16px;
                    padding:1.8rem;text-align:center;height:180px;'>
            <div style='font-size:2.2rem;margin-bottom:0.8rem;'>{icon}</div>
            <h4 style='color:white;margin:0 0 0.5rem 0;font-size:1rem;'>{title}</h4>
            <p style='color:#64748B;font-size:0.85rem;margin:0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0D1B2E;border:1px solid #02C39A;border-radius:12px;
                padding:1.5rem 2rem;margin-bottom:1.5rem;'>
        <h3 style='color:#02C39A;margin:0 0 0.8rem 0;'>✅ Formats de données acceptés</h3>
        <div style='display:flex;gap:2rem;flex-wrap:wrap;'>
            <div style='color:#CBD5E1;'>
                <strong style='color:white;'>📁 Fichier</strong><br>
                CSV encodé UTF-8 ou Latin-1
            </div>
            <div style='color:#CBD5E1;'>
                <strong style='color:white;'>🏭 Secteurs</strong><br>
                Télécom · Fitness · E-commerce · EdTech · SaaS B2B
            </div>
            <div style='color:#CBD5E1;'>
                <strong style='color:white;'>📋 Colonnes requises</strong><br>
                Ancienneté (tenure) + Charges + Colonne Churn (0/1)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        if st.button("📤 Importer mes données maintenant", use_container_width=True, type="primary"):
            st.session_state["_nav_override"] = "📤 Importer mes données"
            st.rerun()  # Le radio sera initialisé à l'index 1 au prochain run

    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;
                padding:1.2rem 1.5rem;margin-top:1rem;'>
        <p style='color:#64748B;font-size:0.85rem;margin:0;'>
            💡 <strong style='color:#94A3B8;'>Conseil :</strong>
            Pour de meilleurs résultats, utilisez un fichier d'au moins 500 lignes.
            Votre modèle est sauvegardé automatiquement et chargé à chaque connexion.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW (original conservé)
# ══════════════════════════════════════════════════════════════════
elif section == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;font-size:3rem;color:white;'>🔮 RetainIQ — Churn Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-container">
            <h2 style='margin:0;font-size:2.5rem;'>{df.shape[0]:,}</h2>
            <p style='margin:5px 0 0 0;font-size:1.1rem;'>Total Clients</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        churn_rate = (df['Churn'].sum() / len(df)) * 100
        st.markdown(f"""<div class="metric-container">
            <h2 style='margin:0;font-size:2.5rem;'>{churn_rate:.1f}%</h2>
            <p style='margin:5px 0 0 0;font-size:1.1rem;'>Taux de Churn</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-container">
            <h2 style='margin:0;font-size:2.5rem;'>{model_accuracy:.1%}</h2>
            <p style='margin:5px 0 0 0;font-size:1.1rem;'>Précision Modèle</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        n_urgent = len(df[df['ChurnProba'] > 0.6])
        st.markdown(f"""<div class="metric-container">
            <h2 style='margin:0;font-size:2.5rem;'>{n_urgent}</h2>
            <p style='margin:5px 0 0 0;font-size:1.1rem;'>🔴 Clients Urgents</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Profils clients avec score de risque IA")

    sample_df      = df.head(10).copy()
    sample_features= sample_df.drop(['Churn', 'ChurnProba', 'RiskLevel'], axis=1, errors='ignore')
    risk_scores    = model.predict_proba(prepare_features_for_prediction(sample_features, feature_names))[:, 1]

    _ov_tenure_col  = next((c for c in ['tenure', 'anciennete_mois', 'mois_inscrit', 'mois_client'] if c in sample_df.columns), None)
    _ov_charges_col = next((c for c in ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen'] if c in sample_df.columns), None)
    display_dict = {
        'Client N°':        range(1, 11),
        cfg['churn_label']: ['Oui' if x == 1 else 'Non' for x in sample_df['Churn'].values],
        'Score de risque':  [f"{x:.1%}" for x in risk_scores],
        'Niveau':           ['🔴 Élevé' if x > 0.6 else '🟡 Modéré' if x > 0.3 else '🟢 Faible' for x in risk_scores]
    }
    if _ov_tenure_col:
        display_dict[cfg['tenure']] = sample_df[_ov_tenure_col].values
    if _ov_charges_col:
        display_dict[cfg['charges']] = sample_df[_ov_charges_col].round(2).values
    if 'TotalCharges' in sample_df.columns:
        display_dict['Total cumulé (€)'] = sample_df['TotalCharges'].round(2).values
    if 'SeniorCitizen' in sample_df.columns:
        display_dict['Senior'] = ['Oui' if x == 1 else 'Non' for x in sample_df['SeniorCitizen'].values]
    display_df = pd.DataFrame(display_dict)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = df.drop(['ChurnProba', 'RiskLevel'], axis=1, errors='ignore').to_csv(index=False)
    st.download_button(
        label="📥 Télécharger le dataset complet",
        data=csv_data,
        file_name=f"RetainIQ_Dataset_{df.shape[0]}_clients.csv",
        mime='text/csv',
        help=f"Télécharger les {df.shape[0]} clients avec {len(feature_names)} features"
    )

    if is_real_data:
        st.info("📈 Données réelles Telco Customer Churn (Kaggle)")
    else:
        st.info("🎲 Données simulées — placez 'Telco-Customer-Churn.csv' dans le dossier pour les vraies données")

    st.markdown("""
    <div style='text-align:center;margin:2rem 0;padding:1rem;
               background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);
               border-radius:15px;color:white;'>
        <h4 style='margin:0;'>🔮 RetainIQ — Projet Industriel 2024-2025 · Équipe de 4</h4>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — VISUAL ANALYTICS (original conservé)
# ══════════════════════════════════════════════════════════════════
elif section == "📊 Visual Analytics":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>📊 Advanced Visual Analytics</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Deep Insights into Customer Behavior Patterns</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=[len(df) - df['Churn'].sum(), df['Churn'].sum()],
            names=['Active Customers', 'Churned Customers'],
            title="🎯 Customer Retention Overview",
            color_discrete_sequence=['#00CC96', '#EF553B'], hole=0.4
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        _va_charges_col = next((c for c in ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen'] if c in df.columns), None)
        if _va_charges_col:
            fig = px.histogram(
                df, x=_va_charges_col, color='Churn',
                title="💰 Monthly Charges vs Churn Risk",
                nbins=30, color_discrete_sequence=['#00CC96', '#EF553B']
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Histogramme des charges indisponible : aucune colonne de charges détectée dans ce dataset.")

    _va_tenure_col = next((c for c in ['tenure', 'anciennete_mois', 'mois_inscrit', 'mois_client'] if c in df.columns), None)
    if _va_tenure_col:
        fig = px.box(
            df, x='Churn', y=_va_tenure_col, title="⏰ Customer Tenure Analysis",
            color='Churn', color_discrete_sequence=['#00CC96', '#EF553B'],
            labels={'Churn': 'Customer Status', _va_tenure_col: 'Months with Company'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Active', 'Churned'])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Analyse d'ancienneté (box plot) indisponible : aucune colonne d'ancienneté détectée dans ce dataset.")

    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})\
        .sort_values('Importance', ascending=False).head(10)
    fig = px.bar(
        importance_df, x='Importance', y='Feature',
        title="🔍 AI Model's Top Predictive Features",
        color='Importance', color_continuous_scale='Viridis', orientation='h'
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Churn', data=df, palette=['#4CAF50', '#F44336'], ax=ax)
    ax.set_title('Customer Churn Distribution', pad=20)
    ax.set_xlabel('Churn Status'); ax.set_ylabel('Count')
    st.pyplot(fig)

    if _va_tenure_col:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x=_va_tenure_col, hue='Churn', multiple='stack', bins=30,
                     palette=['#4CAF50', '#F44336'], ax=ax)
        ax.set_title('Customer Tenure vs Churn', pad=20)
        ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.info("ℹ️ Histogramme d'ancienneté indisponible : aucune colonne d'ancienneté détectée dans ce dataset.")

    numeric_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != 'ChurnProba']
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Matrix', pad=20)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — AI PREDICTION (original + jauge ajoutée)
# ══════════════════════════════════════════════════════════════════
elif section == "🔮 AI Prediction":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🔮 AI Churn Prediction Engine</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Get Instant Churn Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)


    cat_sel_mi = {}  # sélections catégorielles pour le mode Manual Input (datasets non-Telco)

    # Detect which columns are available for display/input
    _telco_cols = {'tenure', 'MonthlyCharges', 'TotalCharges'}
    _has_telco  = _telco_cols.issubset(set(feature_names))

    inputs = {}
    if _has_telco:
        # Original Telco fixed-field layout
        st.markdown("""
        <div style="background:#34495E;padding:15px;border-radius:10px;margin:15px 0;">
            <h4>💡 Tip:</h4>
            <p>Focus on these key features for accurate predictions:</p>
            <ul>
                <li>Tenure (months with company)</li>
                <li>Monthly/Total Charges</li>
                <li>Contract Type</li>
                <li>Internet Service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Information")
            inputs['tenure']         = st.slider("Tenure (months)", 1, 100, 24)
            inputs['MonthlyCharges'] = st.slider("Monthly Charges ($)", float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()), 65.0)
            inputs['TotalCharges']   = st.slider("Total Charges ($)", float(df['TotalCharges'].min()), float(df['TotalCharges'].max()), 2000.0)
        with col2:
            st.subheader("Service Information")
            inputs['Contract']        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            inputs['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            inputs['OnlineSecurity']  = st.checkbox("Online Security")
            inputs['TechSupport']     = st.checkbox("Tech Support")
    else:
        # Dynamic layout for custom datasets
        st.info("Adjust feature values to generate a prediction.")
        # Sélectboxes propres pour les variables catégorielles connues
        _active_cat_groups_mi = _get_active_cat_groups(feature_names)
        _dummy_cols_skip_mi   = {c for g in _active_cat_groups_mi.values() for c in g["dummy_cols"]}
        for grp_key, grp_info in _active_cat_groups_mi.items():
            cat_sel_mi[grp_key] = st.selectbox(
                grp_info["label"], grp_info["options"], key=f"mi_cat_{grp_key}"
            )
        # Contrôles pour les autres features (dummy cols masquées)
        col1, col2 = st.columns(2)
        _non_cat_feats = [f for f in feature_names if f not in _dummy_cols_skip_mi]
        for i, feature in enumerate(_non_cat_feats):
            with col1 if i % 2 == 0 else col2:
                if df[feature].nunique() <= 2:
                    inputs[feature] = 1 if st.checkbox(feature) else 0
                else:
                    _fmin  = float(df[feature].min())
                    _fmax  = float(df[feature].max())
                    _fmed  = float(df[feature].median())
                    inputs[feature] = st.slider(feature, _fmin, _fmax, _fmed)

    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        if _has_telco:
            inputs['Contract_One year']           = 1 if inputs['Contract'] == "One year" else 0
            inputs['Contract_Two year']           = 1 if inputs['Contract'] == "Two year" else 0
            inputs['Contract_One_year']           = inputs['Contract_One year']
            inputs['Contract_Two_year']           = inputs['Contract_Two year']
            inputs['InternetService_Fiber optic'] = 1 if inputs['InternetService'] == "Fiber optic" else 0
            inputs['InternetService_Fiber_optic'] = inputs['InternetService_Fiber optic']
            inputs['InternetService_No']          = 1 if inputs['InternetService'] == "No" else 0
            inputs['OnlineSecurity_Yes']          = 1 if inputs['OnlineSecurity'] else 0
            inputs['TechSupport_Yes']             = 1 if inputs['TechSupport'] else 0
            for key in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']:
                if key in inputs: del inputs[key]
        else:
            # Dataset custom : traduire les selectboxes catégorielles en colonnes dummifiées
            for grp_key, selected_option in cat_sel_mi.items():
                grp_info = CATEGORICAL_DUMMIES_MAP[grp_key]
                for col, val in zip(grp_info["dummy_cols"], grp_info["encoding"][selected_option]):
                    inputs[col] = val

        final_inputs = {feature: inputs.get(feature, 0) for feature in feature_names}
        input_df     = pd.DataFrame([final_inputs])
        prediction   = model.predict_proba(prepare_features_for_prediction(input_df, feature_names))[0][1]

        # Jauge visuelle (NOUVEAU)
        fig_g, lbl, col_g, css = risk_gauge(prediction)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(fig_g, use_container_width=True)
        with c2:
            st.markdown(f"""
            <div class='{css}'>
                <div style='font-size:1.5rem;font-weight:700;color:{col_g};'>{lbl}</div>
                <div style='font-size:3rem;font-weight:800;color:{col_g};margin:10px 0;'>{prediction*100:.1f}%</div>
                <div style='color:#94A3B8;font-size:0.9rem;'>Probabilité de départ du client</div>
            </div>
            """, unsafe_allow_html=True)

        if prediction > 0.6:
            st.markdown("""
            <div style="background:#664d03;padding:15px;border-radius:10px;color:white;">
                <h4>🚨 Retention Recommendations:</h4>
                <ul>
                    <li>Offer loyalty discount or special promotion</li>
                    <li>Provide personalized service check-in</li>
                    <li>Consider contract renewal incentives</li>
                    <li>Address any service issues proactively</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#084298;padding:15px;border-radius:10px;color:white;">
                <h4>💡 Engagement Suggestions:</h4>
                <ul>
                    <li>Continue providing excellent service</li>
                    <li>Consider upselling additional services</li>
                    <li>Check-in periodically to maintain satisfaction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — FUTURE SCENARIOS (original conservé)
# ══════════════════════════════════════════════════════════════════
elif section == "🌟 Future Scenarios":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🌟 Future Impact Simulator</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Predict How Changes Will Impact Customer Churn</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Price Changes")
        price_change = st.slider("Monthly Charges Change (%)", -50, 100, 0)
    with col2:
        st.subheader("⏰ Market Conditions")
        tenure_impact = st.slider("Average Tenure Impact (%)", -50, 50, 0)

    if st.button("🚀 Run Scenario Analysis", type="primary"):
        df_scenario = df.copy()
        if 'MonthlyCharges' in df_scenario.columns:
            df_scenario['MonthlyCharges'] *= (1 + price_change / 100)
        if 'tenure' in df_scenario.columns:
            df_scenario['tenure'] *= (1 + tenure_impact / 100)
            df_scenario['tenure'] = df_scenario['tenure'].clip(1, 100)
        if all(c in df_scenario.columns for c in ['MonthlyCharges', 'tenure', 'TotalCharges']):
            df_scenario['TotalCharges'] = df_scenario['MonthlyCharges'] * df_scenario['tenure']

        X_scenario    = df_scenario.drop(["Churn", "ChurnProba", "RiskLevel"], axis=1, errors='ignore')
        X_current     = df.drop(["Churn", "ChurnProba", "RiskLevel"], axis=1, errors='ignore')
        future_probas = model.predict_proba(prepare_features_for_prediction(X_scenario, feature_names))[:, 1]
        current_probas= model.predict_proba(prepare_features_for_prediction(X_current,  feature_names))[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="metric-container">
                <h2 style='margin:0;font-size:2rem;'>{current_probas.mean():.1%}</h2>
                <p style='margin:5px 0 0 0;font-size:1.1rem;'>Current Churn Risk</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            change_indicator = "📈" if future_probas.mean() > current_probas.mean() else "📉"
            risk_change = ((future_probas.mean() - current_probas.mean()) / current_probas.mean()) * 100
            st.markdown(f"""<div class="metric-container">
                <h2 style='margin:0;font-size:2rem;'>{future_probas.mean():.1%} {change_indicator}</h2>
                <p style='margin:5px 0 0 0;font-size:1.1rem;'>Future Churn Risk ({risk_change:+.1f}%)</p>
            </div>""", unsafe_allow_html=True)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Current Risk Distribution', 'Future Risk Distribution'))
        fig.add_trace(go.Histogram(x=current_probas, name='Current Risk',
                                   marker_color='lightblue', opacity=0.7, nbinsx=30), row=1, col=1)
        fig.add_trace(go.Histogram(x=future_probas,  name='Future Risk',
                                   marker_color='salmon',    opacity=0.7, nbinsx=30), row=1, col=2)
        fig.update_layout(height=500, showlegend=True,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_text="📊 Scenario Impact Analysis")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — SIMULATEUR WHAT-IF (agnostique au secteur)
# ══════════════════════════════════════════════════════════════════
elif section == "⚡ Simulateur What-If":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>⚡ Simulateur What-If</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Testez l'impact de vos actions avant de les mettre en œuvre</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 Modifiez les paramètres d'un client et voyez instantanément comment son risque de churn évolue.")

    # ── Colonnes disponibles pour la simulation ─────────────────────────────
    _exclude_sim = {'Churn', 'ChurnProba', 'RiskLevel', 'Motif de Risque', 'Action Suggérée', 'Priorité'}
    sim_features = [c for c in feature_names if c in df.columns and c not in _exclude_sim]

    def _whatsif_spec(col):
        """Classify a feature column → ('binary'|'discrete'|'continuous'|'constant', params)."""
        s    = df[col].dropna()
        vals = sorted(s.unique().tolist())
        n    = len(vals)
        if n <= 1:
            return ("constant", float(vals[0]) if vals else 0.0)
        # Binary 0/1
        if n == 2 and abs(float(vals[0])) < 1e-9 and abs(float(vals[1]) - 1.0) < 1e-9:
            return ("binary", None)
        # Small integer set (ordinal / one-hot like)
        if n <= 10 and all(float(v) == int(float(v)) for v in vals):
            return ("discrete", [int(v) for v in vals])
        # Continuous
        return ("continuous", (float(s.min()), float(s.max()), float(s.median())))

    # Pré-calculer les specs une seule fois
    _specs = {c: _whatsif_spec(c) for c in sim_features}

    # Masquer les dummy cols des variables catégorielles connues → selectboxes propres
    _active_cat_groups_wi = _get_active_cat_groups(feature_names)
    _dummy_cols_skip_wi   = {c for g in _active_cat_groups_wi.values() for c in g["dummy_cols"]}
    _sim_feats_filtered   = [f for f in sim_features if f not in _dummy_cols_skip_wi]

    cat_sel_a_wi: dict = {}
    cat_sel_b_wi: dict = {}
    inputs_a: dict = {}
    inputs_b: dict = {}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Situation ACTUELLE")
        for grp_key, grp_info in _active_cat_groups_wi.items():
            cat_sel_a_wi[grp_key] = st.selectbox(
                grp_info["label"], grp_info["options"], key=f"wa_cat_{grp_key}"
            )
        for feat in _sim_feats_filtered:
            kind, params = _specs[feat]
            if kind == "constant":
                inputs_a[feat] = params
            elif kind == "binary":
                opts = ["Non (0)", "Oui (1)"]
                sel  = st.selectbox(feat, opts, index=0, key=f"wa_{feat}")
                inputs_a[feat] = 0 if sel.startswith("Non") else 1
            elif kind == "discrete":
                sel  = st.selectbox(feat, params, index=0, key=f"wa_{feat}")
                inputs_a[feat] = int(sel)
            else:
                fmin, fmax, fmed = params
                if fmin >= fmax:
                    inputs_a[feat] = fmed
                    st.number_input(feat, value=fmed, disabled=True, key=f"wa_{feat}")
                else:
                    inputs_a[feat] = st.slider(feat, fmin, fmax, fmed, key=f"wa_{feat}")

    with col2:
        st.subheader("🎯 Situation APRÈS votre action")
        for grp_key, grp_info in _active_cat_groups_wi.items():
            cat_sel_b_wi[grp_key] = st.selectbox(
                grp_info["label"], grp_info["options"], key=f"wb_cat_{grp_key}"
            )
        for feat in _sim_feats_filtered:
            kind, params = _specs[feat]
            if kind == "constant":
                inputs_b[feat] = params
            elif kind == "binary":
                opts     = ["Non (0)", "Oui (1)"]
                def_idx  = 0
                sel      = st.selectbox(feat, opts, index=def_idx, key=f"wb_{feat}")
                inputs_b[feat] = 0 if sel.startswith("Non") else 1
            elif kind == "discrete":
                def_idx  = 0
                sel      = st.selectbox(feat, params, index=def_idx, key=f"wb_{feat}")
                inputs_b[feat] = int(sel)
            else:
                fmin, fmax, fmed = params
                if fmin >= fmax:
                    inputs_b[feat] = fmed
                    st.number_input(feat, value=fmed, disabled=True, key=f"wb_{feat}")
                else:
                    inputs_b[feat] = st.slider(feat, fmin, fmax, fmed, key=f"wb_{feat}")

    # Expansion des sélections catégorielles en colonnes dummifiées attendues par le modèle
    for grp_key, selected_option in cat_sel_a_wi.items():
        grp_info = CATEGORICAL_DUMMIES_MAP[grp_key]
        for col, val in zip(grp_info["dummy_cols"], grp_info["encoding"][selected_option]):
            inputs_a[col] = val
    for grp_key, selected_option in cat_sel_b_wi.items():
        grp_info = CATEGORICAL_DUMMIES_MAP[grp_key]
        for col, val in zip(grp_info["dummy_cols"], grp_info["encoding"][selected_option]):
            inputs_b[col] = val

    # ── Construction des DataFrames de prédiction ───────────────────────────
    _row_a = {f: inputs_a.get(f, float(df[f].median()) if f in df.columns else 0) for f in feature_names}
    _row_b = {f: inputs_b.get(f, float(df[f].median()) if f in df.columns else 0) for f in feature_names}

    score_a = model.predict_proba(prepare_features_for_prediction(pd.DataFrame([_row_a]), feature_names))[0][1]
    score_b = model.predict_proba(prepare_features_for_prediction(pd.DataFrame([_row_b]), feature_names))[0][1]
    delta   = score_b - score_a

    st.markdown("---")
    st.subheader("📊 Résultat de la simulation")
    c1, c2, c3 = st.columns(3)

    with c1:
        fg_a, la, ca, _ = risk_gauge(score_a)
        st.markdown("**Avant l'action**")
        st.plotly_chart(fg_a, use_container_width=True, key="gauge_avant")
        st.markdown(f"<div style='text-align:center;color:{ca};font-weight:700;font-size:1.2rem;'>{score_a*100:.1f}% — {la}</div>", unsafe_allow_html=True)

    with c3:
        fg_b, lb, cb, _ = risk_gauge(score_b)
        st.markdown("**Après l'action**")
        st.plotly_chart(fg_b, use_container_width=True, key="gauge_apres")
        st.markdown(f"<div style='text-align:center;color:{cb};font-weight:700;font-size:1.2rem;'>{score_b*100:.1f}% — {lb}</div>", unsafe_allow_html=True)

    with c2:
        dc = "#00CC96" if delta < 0 else "#EF4444"
        di = "📉 Amélioration" if delta < 0 else "📈 Dégradation"
        # Économie mensuelle si une colonne de charges est disponible
        _charges_cols = ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen']
        _chg_col      = next((c for c in _charges_cols if c in inputs_a), None)
        _savings_html = ""
        if _chg_col:
            _savings = float(inputs_a.get(_chg_col, 0)) - float(inputs_b.get(_chg_col, 0))
            if _savings != 0:
                _savings_html = (
                    f"<div style='color:#888;font-size:0.8rem;margin-top:12px;'>"
                    f"Économie client : <b style='color:#00CC96;'>{_savings:.0f}€/mois</b></div>"
                )
        st.markdown(f"""
        <div class='section-card' style='text-align:center;margin-top:55px;'>
            <div style='font-size:0.9rem;color:#888;'>Impact de l'action</div>
            <div style='font-size:2.8rem;font-weight:800;color:{dc};'>{delta*100:+.1f}%</div>
            <div style='color:{dc};font-weight:600;margin-top:6px;'>{di}</div>
            {_savings_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Recommandations pour ce client")
    _tenure_cols  = ['tenure', 'anciennete_mois', 'mois_inscrit', 'mois_client']
    _charges_cols = ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen']
    _ten_val = float(inputs_a.get(next((c for c in _tenure_cols  if c in inputs_a), ""), 24) or 24)
    _chg_val = float(inputs_a.get(next((c for c in _charges_cols if c in inputs_a), ""), 65) or 65)
    recs = get_recommendations(score_a, _ten_val, _chg_val)
    cols = st.columns(3)
    for i, rec in enumerate(recs):
        cols[i].markdown(f"<div class='section-card'><p style='color:#CBD5E1;margin:0;font-size:0.9rem;'>{rec}</p></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — ALERTES CLIENTS (NOUVEAU)
# ══════════════════════════════════════════════════════════════════
elif section == "🚨 Alertes Clients":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🚨 Alertes Clients à Risque</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Liste priorisée des clients nécessitant une action immédiate</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        seuil = st.slider("Seuil de risque minimum (%)", 30, 90, 60, step=5)
    with col2:
        tri = st.selectbox("Trier par", ["Score décroissant", "Charges décroissantes", "Ancienneté croissante"])

    clients_risque = df[df['ChurnProba'] > seuil / 100].copy()
    _al_charges_col = next((c for c in ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen'] if c in clients_risque.columns), None)
    _al_tenure_col  = next((c for c in ['tenure', 'anciennete_mois', 'mois_inscrit', 'mois_client'] if c in clients_risque.columns), None)
    if tri == "Score décroissant":
        clients_risque = clients_risque.sort_values('ChurnProba', ascending=False)
    elif tri == "Charges décroissantes":
        if _al_charges_col:
            clients_risque = clients_risque.sort_values(_al_charges_col, ascending=False)
        else:
            clients_risque = clients_risque.sort_values('ChurnProba', ascending=False)
            st.info("ℹ️ Tri par charges indisponible pour ce dataset — tri par score appliqué.")
    else:
        if _al_tenure_col:
            clients_risque = clients_risque.sort_values(_al_tenure_col, ascending=True)
        else:
            clients_risque = clients_risque.sort_values('ChurnProba', ascending=False)
            st.info("ℹ️ Tri par ancienneté indisponible pour ce dataset — tri par score appliqué.")

    # ── Moteur de triage statistique ────────────────────────────────
    clients_risque = triage_risque(clients_risque, df)

    ca, cb, cc = st.columns(3)
    ca.markdown(f"""<div class="metric-container"><h2 style='margin:0;font-size:2rem;'>{len(clients_risque)}</h2><p>Clients à risque >{seuil}%</p></div>""", unsafe_allow_html=True)
    cb.markdown(f"""<div class="metric-container"><h2 style='margin:0;font-size:2rem;'>{clients_risque['ChurnProba'].mean()*100:.1f}%</h2><p>Score moyen du groupe</p></div>""", unsafe_allow_html=True)
    if _al_charges_col:
        cc.markdown(f"""<div class="metric-container"><h2 style='margin:0;font-size:2rem;'>{clients_risque[_al_charges_col].sum():.0f}€</h2><p>Revenu mensuel à risque</p></div>""", unsafe_allow_html=True)
    else:
        cc.markdown("""<div class="metric-container"><h2 style='margin:0;font-size:2rem;'>N/A</h2><p>Revenu mensuel à risque<br><small style='opacity:0.7;'>Colonne non disponible</small></p></div>""", unsafe_allow_html=True)

    # ── Filtre par Motif de Risque ───────────────────────────────────
    st.markdown("---")
    motifs_disponibles = ["Tous les motifs"] + sorted(clients_risque['Motif de Risque'].unique().tolist())
    motif_selectionne = st.selectbox("🔍 Filtrer par Motif de Risque", motifs_disponibles, key="filtre_motif")

    if motif_selectionne == "Tous les motifs":
        clients_filtres = clients_risque
    else:
        clients_filtres = clients_risque[clients_risque['Motif de Risque'] == motif_selectionne]

    # ── Tableau filtré ───────────────────────────────────────────────
    show_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'ChurnProba', 'RiskLevel'] if c in clients_filtres.columns]
    show_cols += ['Motif de Risque', 'Action Suggérée']

    disp = clients_filtres[show_cols].head(50).copy()
    disp.index = range(1, len(disp) + 1)
    disp['ChurnProba'] = disp['ChurnProba'].apply(lambda x: f"{x*100:.1f}%")
    disp = disp.rename(columns={
        'tenure': cfg['tenure'], 'MonthlyCharges': cfg['charges'],
        'TotalCharges': 'Total (€)', 'SeniorCitizen': 'Senior',
        'ChurnProba': '⚠️ Score', 'RiskLevel': 'Niveau'
    })
    st.dataframe(disp, use_container_width=True)

    # ── Export Excel ciblé ───────────────────────────────────────────
    def _to_excel(dataframe: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df = dataframe.drop(columns=['ChurnProba', 'RiskLevel'], errors='ignore')
            export_df.to_excel(writer, index=False, sheet_name='Alertes')
            workbook  = writer.book
            worksheet = writer.sheets['Alertes']
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#667eea', 'font_color': 'white', 'border': 1})
            for col_num, col_name in enumerate(export_df.columns):
                worksheet.write(0, col_num, col_name, header_fmt)
                worksheet.set_column(col_num, col_num, max(18, len(str(col_name)) + 4))
        return output.getvalue()

    label_motif = motif_selectionne.replace(' ', '_').replace("'", '').replace('/', '-')[:30]
    excel_bytes = _to_excel(clients_filtres)
    st.download_button(
        label=f"📥 Exporter {len(clients_filtres)} clients — {motif_selectionne} (Excel)",
        data=excel_bytes,
        file_name=f"alertes_{seuil}pct_{label_motif}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="export_excel_alertes",
    )

    # Bloc rapport hebdomadaire — à placer juste sous la liste des clients à risque
    st.markdown("---")
    st.subheader("📧 Rapport hebdomadaire")

    recipient_email = st.text_input("Email destinataire", value=st.session_state.get("user_email", ""), key="weekly_report_email")
    if st.button("Générer et envoyer le rapport PDF", key="send_weekly_report"):
        if not recipient_email.strip():
            st.error("Veuillez saisir un email destinataire.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, "retainiq_report.pdf")

                generate_pdf_report(
                    df=clients_risque if len(clients_risque) > 0 else df,
                    company_name=user_company,
                    sector=secteur,
                    output_path=pdf_path,
                    report_title="Rapport RetainIQ - Clients à risque",
                )

                ok = send_pdf_via_sendgrid(
                    to_email=recipient_email,
                    subject=f"RetainIQ - Rapport clients à risque - {user_company}",
                    body_text="Bonjour,\n\nVeuillez trouver en pièce jointe votre rapport RetainIQ.\n\nCordialement,\nRetainIQ",
                    pdf_path=pdf_path,
                    from_email=os.getenv("SENDER_EMAIL"),
                    from_name=os.getenv("SENDER_NAME", "RetainIQ"),
                )

                if ok:
                    st.success("Rapport envoyé avec succès.")
                else:
                    st.error("Envoi échoué.")

    st.markdown("---")
    st.subheader("📌 Actions recommandées pour ce groupe")
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='alert-box'>📞 <b>Appel de rétention</b><br>Contacter prioritairement les clients avec score >80% dans les 48h</div>", unsafe_allow_html=True)
    c2.markdown("<div class='alert-box'>🎁 <b>Offre personnalisée</b><br>Remise ciblée ou upgrade selon le profil de chaque client</div>", unsafe_allow_html=True)
    c3.markdown("<div class='alert-box'>📧 <b>Campagne email</b><br>Enquête de satisfaction dans les 24h pour identifier les blocages</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 7 — ASSISTANT IA CHATBOT (NOUVEAU)
# ══════════════════════════════════════════════════════════════════
elif section == "🤖 Assistant IA":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>🤖 Assistant IA RetainIQ</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>Posez vos questions sur le churn, les clients et le modèle</p>
    </div>
    """, unsafe_allow_html=True)

    n_urgent_init = int((df["ChurnProba"] > 0.6).sum()) if "ChurnProba" in df.columns else 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "text": (
                    f"Bonjour ! Je suis l'Assistant IA de RetainIQ, propulsé par Gemini 🔮\n\n"
                    f"Je peux vous aider sur :\n"
                    f"- Les **risques de churn** de vos clients\n"
                    f"- Les **causes principales** de départ\n"
                    f"- Les **actions de rétention** recommandées\n"
                    f"- Le **fonctionnement du modèle XGBoost**\n\n"
                    f"Actuellement **{n_urgent_init} clients** sont en risque élevé. Que voulez-vous savoir ?"
                ),
            }
        ]

    # Render conversation history
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["text"])

    # Quick-question buttons (rendered before chat_input so they sit above the input bar)
    st.markdown("---")
    st.caption("💬 Questions rapides")
    quick_qs = [
        "Pourquoi les clients partent-ils ?",
        "Comment fonctionne XGBoost ?",
        "Que faire pour retenir un client ?",
        "Combien de clients sont urgents ?",
    ]
    cols = st.columns(len(quick_qs))
    for i, q in enumerate(quick_qs):
        if cols[i].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": q})
            with st.spinner("L'IA analyse..."):
                reply = gemini_chat_response(q, df)
            st.session_state.chat_history.append({"role": "assistant", "text": reply})
            st.rerun()

    if st.button("🗑️ Effacer la conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Main chat input 
    user_input = st.chat_input("Posez votre question à l'Assistant IA…")
    if user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("L'IA analyse..."):
                reply = gemini_chat_response(user_input, df)
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "text": reply})
        st.rerun()

elif section == "📤 Importer mes données":
    show_pipeline_page(
        user_email=st.session_state.get("user_email", "default"),
        secteur=secteur
    )
elif section == "🧠 Explainable AI":
    show_shap_page(model, df, feature_names)

elif section == "📧 Campagnes & Rapports":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;color:white;'>📧 Campagnes & Rapports</h1>
        <p style='margin:10px 0 0 0;opacity:0.9;color:white;'>
            CRM intelligent — Rapports planifiés, relances ciblées et campagnes occasionnelles
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Mes Rapports Planifiés",
        "🎯 Relance (Smart Rétention)",
        "📢 Occasions & Fêtes",
        "📜 Historique CRM",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Rapports Planifiés (code original intégralement préservé)
    # ══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## ⏰ Rapports Hebdomadaires Planifiés")
        st.markdown("Gérez l'envoi automatique des rapports PDF à tous les utilisateurs enregistrés.")
        st.markdown("---")

        status = sched.get_status()

        # ── Statut actuel ──────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            if status["running"]:
                st.success("🟢 Scheduler actif")
            else:
                st.error("🔴 Scheduler inactif")
        with col2:
            st.metric("Prochaine exécution", status["next_run"])
        with col3:
            st.metric("Jobs enregistrés", status["job_count"])

        st.markdown("---")

        # ── Configuration de la diffusion ─────────────────────────────
        st.markdown("### 📧 Configuration de la diffusion")
        st.caption(
            "Ces destinataires recevront les alertes de churn filtrées par motif de risque, "
            "selon la planification définie ci-dessous."
        )

        default_emails = st.session_state.get(
            "report_recipients",
            "manager1@entreprise.com, manager2@entreprise.com",
        )
        recipients_input = st.text_area(
            "Adresses email des destinataires",
            value=default_emails,
            placeholder="manager1@entreprise.com, manager2@entreprise.com",
            help="Saisissez les adresses email séparées par des virgules.",
            key="recipients_textarea",
        )

        if st.button("💾 Enregistrer la liste des managers", use_container_width=True):
            st.session_state["report_recipients"] = recipients_input.strip()
            emails_list = [e.strip() for e in recipients_input.split(",") if e.strip()]
            st.success(
                f"✅ Configuration sauvegardée — {len(emails_list)} destinataire(s) enregistré(s). "
                "Ils recevront les alertes filtrées par motif de risque selon la planification ci-dessous."
            )

        st.markdown("---")

        # ── Configuration de la planification ─────────────────────────
        st.markdown("### ⚙️ Configurer la planification")

        JOURS = {
            "Lundi": "mon", "Mardi": "tue", "Mercredi": "wed",
            "Jeudi": "thu", "Vendredi": "fri", "Samedi": "sat", "Dimanche": "sun",
        }

        jours_selectionnes = st.multiselect(
            "Jours d'envoi",
            options=list(JOURS.keys()),
            default=["Lundi"],
            key="sched_days",
        )
        heure_envoi = st.time_input(
            "Heure d'envoi du rapport",
            value=datetime.time(8, 0),
            key="sched_time",
        )

        if st.button("💾 Enregistrer la planification", use_container_width=True):
            if not jours_selectionnes:
                st.warning("⚠️ Veuillez sélectionner au moins un jour d'envoi.")
            else:
                jours_cron = ",".join(JOURS[j] for j in jours_selectionnes)
                sched.update_schedule(
                    day_of_week=jours_cron,
                    hour=heure_envoi.hour,
                    minute=heure_envoi.minute,
                )
                jours_str = ", ".join(jours_selectionnes)
                st.success(
                    f"✅ Planification mise à jour : chaque {jours_str} "
                    f"à {heure_envoi.strftime('%H:%M')}."
                )
                st.rerun()

        st.markdown("---")

        # ── Envoi manuel ───────────────────────────────────────────────
        st.markdown("### 🚀 Envoi manuel immédiat")
        st.info("Déclenche l'envoi des rapports maintenant, sans attendre la planification.")

        if st.button("📨 Envoyer les rapports maintenant", use_container_width=True, type="primary"):
            with st.spinner("Génération et envoi des rapports en cours…"):
                try:
                    sched.trigger_now()
                    st.success("✅ Rapports envoyés avec succès à tous les utilisateurs.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'envoi : {e}")
            st.rerun()

        st.markdown("---")

        # ── Historique des exécutions ──────────────────────────────────
        st.markdown("### 📋 Historique des exécutions")

        history = status["history"]
        if not history:
            st.info("Aucune exécution enregistrée pour cette session.")
        else:
            hist_df = pd.DataFrame(history)[["date", "status", "duration_s", "detail"]]
            hist_df.columns = ["Date", "Statut", "Durée (s)", "Détail"]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Relance Smart Rétention
    # ══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🎯 Relance Intelligente — Clients à Risque")

        _t2_mode = st.radio(
            "Mode d'envoi",
            ["👤 Relance Individuelle (Recherche)", "👥 Relance Groupée (Filtres)"],
            horizontal=True,
            key="t2_mode",
        )
        st.markdown("---")

        # Détection des colonnes — partagée entre les deux modes
        _t2_charges_col = next(
            (c for c in ['MonthlyCharges', 'abonnement_mensuel', 'mrr', 'panier_moyen']
             if c in df.columns), None
        )
        _t2_tenure_col = next(
            (c for c in ['tenure', 'anciennete_mois', 'mois_inscrit', 'mois_client']
             if c in df.columns), None
        )
        _t2_email_col = next(
            (c for c in df.columns
             if any(kw in c.lower() for kw in ['email', 'mail', 'courriel'])),
            None,
        )
        _t2_phone_col = next(
            (c for c in df.columns
             if any(kw in c.lower() for kw in ['phone', 'tel', 'telephone', 'mobile'])),
            None,
        )
        _t2_name_col = next(
            (c for c in df.columns
             if any(kw in c.lower() for kw in ['name', 'nom', 'prenom', 'client_name'])),
            None,
        )

        # ── MODE INDIVIDUEL ────────────────────────────────────────────
        if _t2_mode == "👤 Relance Individuelle (Recherche)":
            st.caption("Sélectionnez un client en danger et envoyez-lui un email de rétention personnalisé.")

            high_risk_df = df[df['ChurnProba'] > 0.6].copy().reset_index(drop=True)
            high_risk_df.index = range(1, len(high_risk_df) + 1)

            if high_risk_df.empty:
                st.info("ℹ️ Aucun client avec un score > 60% dans le dataset actuel.")
            else:
                st.markdown(f"""
                <div class='section-card'>
                    <span style='color:#EF4444;font-weight:700;font-size:1.1rem;'>🔴 {len(high_risk_df)} clients</span>
                    <span style='color:#94A3B8;'> en risque élevé (score &gt; 60%)</span>
                </div>
                """, unsafe_allow_html=True)

                def _client_label(i, row):
                    _email_val = None
                    if _t2_email_col:
                        _raw = row.get(_t2_email_col)
                        if _raw is not None and pd.notna(_raw) and str(_raw).strip():
                            _email_val = str(_raw).strip()
                    identifier = _email_val if _email_val else f"Client N°{i}"
                    parts = [f"{identifier}  —  Score {row['ChurnProba']*100:.0f}%"]
                    if _t2_phone_col:
                        _ph = row.get(_t2_phone_col)
                        if _ph is not None and pd.notna(_ph) and str(_ph).strip():
                            parts.append(f"📞 {str(_ph).strip()}")
                    if _t2_tenure_col:
                        parts.append(f"Ancienneté {row[_t2_tenure_col]:.0f} mois")
                    if _t2_charges_col:
                        parts.append(f"{row[_t2_charges_col]:.0f}€/mois")
                    return "  ·  ".join(parts)

                client_options = {
                    _client_label(i, row): i
                    for i, row in high_risk_df.iterrows()
                }

                selected_label = st.selectbox(
                    "🔍 Choisir un client à relancer",
                    options=list(client_options.keys()),
                    key="t2_client_select",
                )
                selected_idx = client_options[selected_label]
                client_row   = high_risk_df.loc[selected_idx]
                score_val    = float(client_row['ChurnProba'])

                _t2_email_prefill = ""
                if _t2_email_col:
                    _raw_email = client_row.get(_t2_email_col)
                    if _raw_email is not None and pd.notna(_raw_email) and str(_raw_email).strip():
                        _t2_email_prefill = str(_raw_email).strip()

                tenure_disp  = f"{client_row[_t2_tenure_col]:.0f} mois" if _t2_tenure_col else "N/A"
                charges_disp = f"{client_row[_t2_charges_col]:.0f} €/mois" if _t2_charges_col else "N/A"
                c_color      = "#EF4444" if score_val > 0.7 else "#F59E0B"

                st.markdown(f"""
                <div class='section-card' style='margin-top:0.8rem;'>
                    <div style='display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;'>
                        <div>
                            <p style='color:#888;font-size:0.75rem;margin:0;'>Score de churn</p>
                            <p style='color:{c_color};font-size:1.7rem;font-weight:800;margin:0;'>{score_val*100:.1f}%</p>
                        </div>
                        <div>
                            <p style='color:#888;font-size:0.75rem;margin:0;'>Ancienneté</p>
                            <p style='color:white;font-size:1rem;font-weight:600;margin:0;'>{tenure_disp}</p>
                        </div>
                        <div>
                            <p style='color:#888;font-size:0.75rem;margin:0;'>Charges mensuelles</p>
                            <p style='color:white;font-size:1rem;font-weight:600;margin:0;'>{charges_disp}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### 📨 Formulaire d'envoi")

                _t2_to = st.text_input(
                    "📮 Adresse email du destinataire",
                    value=_t2_email_prefill,
                    placeholder="client@exemple.com",
                    key=f"t2_to_{selected_idx}",
                )
                _t2_subject = st.text_input(
                    "Objet de l'email",
                    value=f"Nous tenons à vous garder — Offre exclusive {user_company}",
                    key="t2_subject",
                )

                if "crm_draft_relance" not in st.session_state:
                    st.session_state["crm_draft_relance"] = ""

                if st.button("✨ Générer le texte avec l'IA", key="gen_relance", use_container_width=True):
                    with st.spinner("Gemini rédige votre email de rétention…"):
                        _ctx = (
                            f"Client du secteur {secteur}, ancienneté {tenure_disp}, "
                            f"charges {charges_disp}, score de risque de résiliation {score_val*100:.0f}%. "
                            f"L'objectif est de le fidéliser avec une offre personnalisée attractive."
                        )
                        st.session_state["crm_draft_relance"] = gemini_draft_email(
                            context=_ctx,
                            email_type="relance de rétention personnalisée",
                        )

                _t2_body = st.text_area(
                    "Corps de l'email (modifiable avant envoi)",
                    value=st.session_state.get("crm_draft_relance", ""),
                    height=300,
                    key="t2_body",
                    placeholder="Cliquez sur '✨ Générer le texte avec l'IA' ou rédigez votre message ici…",
                )

                if st.button("📤 Envoyer l'email de relance", type="primary", use_container_width=True, key="send_relance"):
                    if not _t2_to.strip():
                        st.error("Veuillez renseigner l'adresse email du destinataire.")
                    elif not _t2_body.strip():
                        st.error("Le corps de l'email est vide.")
                    else:
                        _html_body = "<p>" + _t2_body.replace("\n", "<br>") + "</p>"
                        with st.spinner("Envoi en cours…"):
                            _ok, _msg = send_campaign_email(_t2_to.strip(), _t2_subject, _html_body)
                        if _ok:
                            st.success(f"✅ Email envoyé à {_t2_to.strip()}")
                            if "crm_history" not in st.session_state:
                                st.session_state["crm_history"] = []
                            st.session_state["crm_history"].append({
                                "date":         datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                                "type":         "Relance Smart",
                                "destinataire": _t2_to.strip(),
                                "objet":        _t2_subject,
                                "statut":       "✅ Envoyé",
                            })
                        else:
                            st.warning(f"⚠️ {_msg}")

        # ── MODE GROUPÉ ────────────────────────────────────────────────
        else:
            st.caption("Appliquez des filtres pour cibler un groupe de clients et envoyez-leur un email en masse.")

            st.markdown("#### 🔧 Filtres de segmentation")
            _fg1, _fg2 = st.columns(2)

            with _fg1:
                _risk_min, _risk_max = st.slider(
                    "🎯 Score de risque (%)",
                    min_value=0, max_value=100,
                    value=(60, 100),
                    step=5,
                    key="t2g_risk_range",
                )
            with _fg2:
                if _t2_tenure_col and df[_t2_tenure_col].notna().any():
                    _ten_min_v = int(df[_t2_tenure_col].min())
                    _ten_max_v = int(df[_t2_tenure_col].max())
                    _ten_range = st.slider(
                        "📅 Ancienneté (mois)",
                        min_value=_ten_min_v, max_value=_ten_max_v,
                        value=(_ten_min_v, _ten_max_v),
                        key="t2g_tenure_range",
                    )
                else:
                    _ten_range = None

            _t2_contract_col = next(
                (c for c in df.columns if 'contract' in c.lower() or 'contrat' in c.lower()),
                None,
            )
            if _t2_contract_col:
                _contract_opts = ["Tous"] + sorted(
                    [str(v) for v in df[_t2_contract_col].dropna().unique()]
                )
                _sel_contract = st.selectbox(
                    f"📄 Type de contrat ({_t2_contract_col})",
                    _contract_opts,
                    key="t2g_contract",
                )
            else:
                _sel_contract = "Tous"

            _mask = (
                (df['ChurnProba'] >= _risk_min / 100) &
                (df['ChurnProba'] <= _risk_max / 100)
            )
            if _ten_range and _t2_tenure_col:
                _mask &= (
                    (df[_t2_tenure_col] >= _ten_range[0]) &
                    (df[_t2_tenure_col] <= _ten_range[1])
                )
            if _sel_contract != "Tous" and _t2_contract_col:
                _mask &= df[_t2_contract_col].astype(str) == _sel_contract

            _filtered_df = df[_mask].copy()

            _preview_cols = []
            if _t2_email_col:
                _preview_cols.append(_t2_email_col)
            if _t2_name_col:
                _preview_cols.append(_t2_name_col)
            _preview_cols.append('ChurnProba')
            _preview_cols.append('RiskLevel')
            if _t2_tenure_col and _t2_tenure_col not in _preview_cols:
                _preview_cols.append(_t2_tenure_col)
            if _t2_charges_col and _t2_charges_col not in _preview_cols:
                _preview_cols.append(_t2_charges_col)
            _preview_cols = [c for c in _preview_cols if c in _filtered_df.columns]

            _no_email_warn = "" if _t2_email_col else " — ⚠️ Aucune colonne email détectée"
            st.markdown(f"""
            <div class='section-card' style='margin-top:0.5rem;'>
                <span style='color:#667eea;font-weight:700;font-size:1.1rem;'>👥 {len(_filtered_df)} clients</span>
                <span style='color:#94A3B8;'> correspondent aux filtres sélectionnés{_no_email_warn}</span>
            </div>
            """, unsafe_allow_html=True)

            if not _filtered_df.empty and _preview_cols:
                with st.expander(f"👁️ Aperçu des {min(len(_filtered_df), 10)} premiers clients filtrés"):
                    st.dataframe(
                        _filtered_df[_preview_cols].head(10).rename(
                            columns={'ChurnProba': 'Score', 'RiskLevel': 'Niveau'}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("---")
            st.markdown("#### 📨 Composition de l'email groupé")

            _t2g_subject = st.text_input(
                "Objet de l'email",
                value=f"Un message important de {user_company}",
                key="t2g_subject",
            )

            if "crm_draft_groupe" not in st.session_state:
                st.session_state["crm_draft_groupe"] = ""

            if st.button("✨ Générer le texte avec l'IA", key="gen_groupe", use_container_width=True):
                with st.spinner("Gemini rédige l'email de groupe…"):
                    _ctx_group = (
                        f"Secteur : {secteur}. Groupe cible : {len(_filtered_df)} clients "
                        f"avec un score de risque entre {_risk_min}% et {_risk_max}%. "
                        f"Ancienneté filtrée : {str(_ten_range) if _ten_range else 'toutes'}. "
                        f"Objectif : fidéliser avec un message engageant. Entreprise : {user_company}."
                    )
                    st.session_state["crm_draft_groupe"] = gemini_draft_email(
                        context=_ctx_group,
                        email_type="relance groupée de rétention",
                    )

            _t2g_body = st.text_area(
                "Corps de l'email (modifiable avant envoi)",
                value=st.session_state.get("crm_draft_groupe", ""),
                height=300,
                key="t2g_body",
                placeholder="Cliquez sur '✨ Générer le texte avec l'IA' ou rédigez votre message ici…",
            )

            if st.button("📤 Envoyer à tous les clients filtrés", type="primary", use_container_width=True, key="send_groupe"):
                if not _t2_email_col:
                    st.error("⛔ Aucune colonne email détectée dans vos données.")
                elif _filtered_df.empty:
                    st.error("Aucun client ne correspond aux filtres sélectionnés.")
                elif not _t2g_body.strip():
                    st.error("Le corps de l'email est vide.")
                else:
                    _emails_groupe = [
                        str(e).strip() for e in _filtered_df[_t2_email_col].dropna()
                        if str(e).strip()
                    ]
                    if not _emails_groupe:
                        st.error("Aucune adresse email valide dans le segment filtré.")
                    else:
                        _html_body_g = "<p>" + _t2g_body.replace("\n", "<br>") + "</p>"
                        _sent_g, _failed_g = 0, 0
                        _prog_g = st.progress(0, text=f"Envoi 0 / {len(_emails_groupe)}…")
                        for _idx_g, _addr_g in enumerate(_emails_groupe):
                            _ok_g, _msg_g = send_campaign_email(_addr_g, _t2g_subject, _html_body_g)
                            if _ok_g:
                                _sent_g += 1
                            else:
                                _failed_g += 1
                            _prog_g.progress(
                                (_idx_g + 1) / len(_emails_groupe),
                                text=f"Envoi {_idx_g + 1} / {len(_emails_groupe)} — {_addr_g}",
                            )
                        _prog_g.empty()
                        if _sent_g > 0:
                            st.success(f"✅ {_sent_g} email(s) envoyé(s) avec succès.")
                            if "crm_history" not in st.session_state:
                                st.session_state["crm_history"] = []
                            st.session_state["crm_history"].append({
                                "date":         datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                                "type":         "Relance Groupée",
                                "destinataire": f"{_sent_g} client(s)",
                                "objet":        _t2g_subject,
                                "statut":       "✅ Envoyé" if _failed_g == 0 else f"⚠️ {_failed_g} échec(s)",
                            })
                        if _failed_g > 0:
                            st.warning(f"⚠️ {_failed_g} envoi(s) échoué(s) — vérifiez BREVO_API_KEY et FROM_EMAIL dans .env")

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Occasions & Fêtes
    # ══════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📢 Campagne Occasion — Envoi Groupé par Segment")
        st.caption("Envoyez un email de vœux ou de promotion à l'occasion d'un événement.")

        _OCCASIONS = [
            "🎉 Bonne Année",
            "🌙 Aïd Moubarak",
            "🌹 Fête des Mères",
            "👨 Fête des Pères",
            "🎄 Joyeux Noël",
            "🇫🇷 Fête Nationale",
            "🎓 Rentrée",
            "🛒 Black Friday",
            "✏️ Occasion personnalisée…",
        ]

        _col_occ1, _col_occ2 = st.columns(2)
        with _col_occ1:
            occasion_type = st.selectbox("🗓️ Type d'occasion", _OCCASIONS, key="t3_occasion")
        with _col_occ2:
            target_segment = st.selectbox(
                "🎯 Segment cible",
                ["Tous les clients", "Risque élevé (>60%)", "Risque modéré (35–60%)", "Clients fidèles (<35%)"],
                key="t3_segment",
            )

        if occasion_type == "✏️ Occasion personnalisée…":
            occasion_label = st.text_input(
                "Décrivez l'occasion",
                placeholder="Ex : Anniversaire 5 ans de l'entreprise",
                key="t3_custom",
            )
        else:
            occasion_label = occasion_type

        # Détection colonne email dans le DataFrame
        _t3_email_col = next(
            (c for c in df.columns
             if any(kw in c.lower() for kw in ['email', 'mail', 'courriel'])),
            None,
        )

        # Extraction automatique des emails selon le segment sélectionné
        if "élevé" in target_segment:
            _t3_seg_df = df[df['ChurnProba'] > 0.6]
        elif "modéré" in target_segment:
            _t3_seg_df = df[(df['ChurnProba'] > 0.35) & (df['ChurnProba'] <= 0.6)]
        elif "fidèles" in target_segment:
            _t3_seg_df = df[df['ChurnProba'] <= 0.35]
        else:
            _t3_seg_df = df

        n_segment = len(_t3_seg_df)

        if _t3_email_col:
            _t3_email_list = [
                str(e).strip() for e in _t3_seg_df[_t3_email_col].dropna()
                if str(e).strip()
            ]
        else:
            _t3_email_list = []

        # Affichage info segment + statut extraction
        if _t3_email_col:
            st.info(
                f"📊 Segment : **{target_segment}** — {n_segment} clients · "
                f"**{len(_t3_email_list)} emails** extraits automatiquement"
            )
        else:
            st.warning(
                "⚠️ Aucune colonne email détectée dans vos données. "
                "Importez un fichier avec une colonne 'email' pour activer l'envoi automatique."
            )

        _t3_subject = st.text_input(
            "Objet de l'email",
            value=f"{occasion_label} — {user_company}",
            key="t3_subject",
        )

        if "crm_draft_occasion" not in st.session_state:
            st.session_state["crm_draft_occasion"] = ""

        if st.button("✨ Générer le texte avec l'IA", key="gen_occasion", use_container_width=True):
            with st.spinner("Gemini rédige votre email de campagne…"):
                _ctx = (
                    f"Occasion : {occasion_label}. Secteur : {secteur}. "
                    f"Segment cible : {target_segment} ({n_segment} clients). "
                    f"Entreprise : {user_company}."
                )
                st.session_state["crm_draft_occasion"] = gemini_draft_email(
                    context=_ctx,
                    email_type="email de vœux ou promotion pour une occasion spéciale",
                )

        _t3_body = st.text_area(
            "Corps de l'email (modifiable avant envoi)",
            value=st.session_state.get("crm_draft_occasion", ""),
            height=300,
            key="t3_body",
            placeholder="Cliquez sur '✨ Générer le texte avec l'IA' ou rédigez votre message ici…",
        )

        if st.button("📤 Envoyer la campagne", type="primary", use_container_width=True, key="send_occasion"):
            if not _t3_email_col:
                st.error("⛔ Aucune colonne email détectée dans vos données.")
            elif not _t3_email_list:
                st.error("Aucune adresse email valide dans ce segment.")
            elif not _t3_body.strip():
                st.error("Le corps de l'email est vide.")
            else:
                _html_body     = "<p>" + _t3_body.replace("\n", "<br>") + "</p>"
                _sent, _failed = 0, 0
                _prog_t3 = st.progress(0, text=f"Envoi 0 / {len(_t3_email_list)}…")
                for _idx_t3, _addr in enumerate(_t3_email_list):
                    _ok, _msg = send_campaign_email(_addr, _t3_subject, _html_body)
                    if _ok:
                        _sent += 1
                    else:
                        _failed += 1
                    _prog_t3.progress(
                        (_idx_t3 + 1) / len(_t3_email_list),
                        text=f"Envoi {_idx_t3 + 1} / {len(_t3_email_list)} — {_addr}",
                    )
                _prog_t3.empty()
                if _sent > 0:
                    st.success(f"✅ {_sent} email(s) envoyé(s) avec succès.")
                    if "crm_history" not in st.session_state:
                        st.session_state["crm_history"] = []
                    st.session_state["crm_history"].append({
                        "date":         datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "type":         f"Campagne {occasion_label}",
                        "destinataire": f"{_sent} destinataire(s)",
                        "objet":        _t3_subject,
                        "statut":       "✅ Envoyé" if _failed == 0 else f"⚠️ {_failed} échec(s)",
                    })
                if _failed > 0:
                    st.warning(f"⚠️ {_failed} envoi(s) échoué(s) — vérifiez BREVO_API_KEY et FROM_EMAIL dans .env")

    # ══════════════════════════════════════════════════════════════
    # TAB 4 — Historique CRM
    # ══════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📜 Historique des Envois CRM")
        st.caption("Suivi de toutes les campagnes et relances envoyées durant cette session.")

        # Données de démonstration pré-chargées au premier accès
        if "crm_history" not in st.session_state:
            st.session_state["crm_history"] = [
                {
                    "date": "28/04/2026 09:15", "type": "Rapport hebdomadaire",
                    "destinataire": "manager@entreprise.com",
                    "objet": "RetainIQ — Rapport semaine 17", "statut": "✅ Envoyé",
                },
                {
                    "date": "25/04/2026 08:00", "type": "Relance Smart",
                    "destinataire": "client.vip@exemple.com",
                    "objet": "Nous tenons à vous garder — Offre exclusive", "statut": "✅ Envoyé",
                },
                {
                    "date": "21/04/2026 10:30", "type": "Rapport hebdomadaire",
                    "destinataire": "direction@entreprise.com",
                    "objet": "RetainIQ — Rapport semaine 16", "statut": "✅ Envoyé",
                },
                {
                    "date": "14/04/2026 08:00", "type": "Campagne 🌙 Aïd Moubarak",
                    "destinataire": "856 destinataires",
                    "objet": "Aïd Moubarak — Offre spéciale pour vous", "statut": "✅ Envoyé",
                },
                {
                    "date": "07/04/2026 08:00", "type": "Rapport hebdomadaire",
                    "destinataire": "manager@entreprise.com",
                    "objet": "RetainIQ — Rapport semaine 14", "statut": "⚠️ 1 échec",
                },
            ]

        _history_records = st.session_state["crm_history"]
        _hist_crm_df = pd.DataFrame(_history_records[::-1])
        _hist_crm_df.columns = ["Date", "Type", "Destinataire(s)", "Objet", "Statut"]
        st.dataframe(_hist_crm_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        _cs1, _cs2, _cs3 = st.columns(3)
        _cs1.metric("📧 Total envois", len(_history_records))
        _cs2.metric("📢 Campagnes",    sum(1 for r in _history_records if "Campagne" in r["type"]))
        _cs3.metric("🎯 Relances",     sum(1 for r in _history_records if "Relance" in r["type"]))


elif section == "🏆 Programme de Fidélité":
    show_loyalty_page(df, secteur, user_company, user_email, user_role)

elif section == "⚙️ Panneau Admin":
    # ── Garde de sécurité côté serveur ──────────────────────────
    if not _is_manager_or_admin:
        st.error("⛔ Accès refusé — réservé aux managers et administrateurs.")
        st.stop()

    from database import (
        get_all_users_admin, get_users_by_company,
        update_user_role, delete_user, VALID_ROLES,
    )
    from auth import register_user

    st.markdown("""
    <div class='main-header'>
        <h1>⚙️ Panneau d'Administration</h1>
        <p>Gestion des utilisateurs et des accès</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tableau des utilisateurs ─────────────────────────────────
    st.markdown("### 👥 Utilisateurs enregistrés")

    # Étanchéité Multi-Tenant : super_admin voit tout, les autres voient uniquement leur entreprise
    if _is_super_admin:
        all_users = get_all_users_admin()
    else:
        all_users = get_users_by_company(user_company)

    _ROLE_COLORS_HEX = {
        "super_admin": "#8B5CF6",
        "admin":       "#EF4444",
        "manager":     "#F59E0B",
        "agent":       "#10B981",
    }
    _ROLE_ICONS = {
        "super_admin": "🔮",
        "admin":       "⚙️",
        "manager":     "📊",
        "agent":       "👤",
    }

    # Options de rôle affichables selon le rang de l'utilisateur connecté
    # Un non-super_admin ne peut ni voir ni attribuer le rôle super_admin
    _assignable_roles = list(VALID_ROLES) if _is_super_admin else [r for r in VALID_ROLES if r != "super_admin"]

    for u in all_users:
        rc = _ROLE_COLORS_HEX.get(u["role"], "#888")
        ri = _ROLE_ICONS.get(u["role"], "👤")
        col_info, col_role, col_del = st.columns([4, 2, 1])
        with col_info:
            st.markdown(f"""
            <div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:8px;
                        padding:10px 14px;margin:4px 0;'>
                <span style='color:white;font-weight:600;'>{u['email']}</span><br>
                <span style='color:#888;font-size:0.8rem;'>{u['company']} · {u['secteur']}</span>
            </div>""", unsafe_allow_html=True)
        with col_role:
            # Si le rôle actuel de l'utilisateur n'est pas dans les options (ex: super_admin vu par un admin),
            # on le force sur le dernier rôle de la liste pour éviter une IndexError.
            _cur_role = u["role"] if u["role"] in _assignable_roles else _assignable_roles[-1]
            new_role = st.selectbox(
                "Rôle",
                options=_assignable_roles,
                index=_assignable_roles.index(_cur_role),
                key=f"role_{u['email']}",
                label_visibility="collapsed",
            )
            if new_role != u["role"]:
                if st.button("💾 Sauvegarder", key=f"save_{u['email']}", use_container_width=True):
                    # Verrou serveur : seul un super_admin peut attribuer le rôle super_admin
                    if new_role == "super_admin" and not _is_super_admin:
                        st.error("⛔ Attribution du rôle super_admin interdite.")
                    else:
                        update_user_role(u["email"], new_role)
                        st.success(f"Rôle de {u['email']} mis à jour → {new_role}")
                        st.rerun()
        with col_del:
            if u["email"] != user_email:  # on ne peut pas se supprimer soi-même
                if st.button("🗑️", key=f"del_{u['email']}", help=f"Supprimer {u['email']}"):
                    delete_user(u["email"])
                    st.warning(f"Compte {u['email']} supprimé.")
                    st.rerun()
            else:
                st.markdown("<p style='color:#555;font-size:0.75rem;text-align:center;padding-top:12px;'>Vous</p>",
                            unsafe_allow_html=True)

    st.markdown("---")
    if _is_super_admin:
        st.markdown("### 📋 Comptes de démonstration")
        st.info("""
    **Comptes pré-créés pour tester le RBAC :**\n
    | Email | Mot de passe | Rôle |
    |---|---|---|
    | super@retainiq.com | SuperAdmin123! | Super Admin |
    | admin@retainiq.com | Admin123! | Admin |
    | manager@retainiq.com | Manager123! | Manager |
    | agent@retainiq.com | Agent123! | Agent |
    """)

    st.markdown("---")

    # ── Gestion de l'équipe (Manager, Admin, Super Admin) ────────
    with st.expander("👥 Gestion de l'équipe", expanded=False):
        if _is_super_admin:
            st.markdown("Créez un compte **admin**, **manager** ou **agent**.")
        elif _is_admin:
            st.markdown("Créez un compte **manager** ou **agent** pour votre entreprise.")
        else:
            st.markdown("Créez un compte **agent** pour votre équipe.")
        with st.form("team_create_agent_form"):
            ta1, ta2 = st.columns(2)
            with ta1:
                agent_email = st.text_input("Email du collaborateur", placeholder="prenom.nom@entreprise.com")
                agent_pwd   = st.text_input("Mot de passe", type="password", key="agent_pwd")
            with ta2:
                if _is_super_admin:
                    # Le super_admin peut créer jusqu'au niveau admin, jamais super_admin
                    agent_role = st.selectbox("Rôle", ["agent", "manager", "admin"], key="agent_role_sel")
                elif _is_admin:
                    agent_role = st.selectbox("Rôle", ["agent", "manager"], key="agent_role_sel")
                else:
                    # Manager : choix forcé sur agent, aucune option affichée
                    agent_role = "agent"
                agent_confirm = st.text_input("Confirmer le mot de passe", type="password", key="agent_confirm")
            agent_submitted = st.form_submit_button("✅ Créer le compte", use_container_width=True, type="primary")

        if agent_submitted:
            if not all([agent_email, agent_pwd, agent_confirm]):
                st.error("Tous les champs sont obligatoires.")
            elif agent_pwd != agent_confirm:
                st.error("Les mots de passe ne correspondent pas.")
            elif len(agent_pwd) < 6:
                st.error("Le mot de passe doit contenir au moins 6 caractères.")
            else:
                ok, msg = register_user(agent_email, agent_pwd, user_company, user_secteur, role=agent_role)
                if ok:
                    st.success(f"✅ Compte {agent_role} créé : {agent_email} (entreprise : {user_company})")
                    st.rerun()
                else:
                    st.error(msg)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem;color:#666;font-size:0.8rem;'>
    🔮 <b>RetainIQ</b> &nbsp;·&nbsp; 
</div>
""", unsafe_allow_html=True)
