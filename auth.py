"""
auth.py — Authentification RetainIQ.

Utilise :
  - SQLite  (via database.py) au lieu de users.json
  - bcrypt  au lieu de SHA256 brut

Migration transparente : les anciens comptes SHA256 sont automatiquement
re-hashés en bcrypt lors de la prochaine connexion réussie.
"""

import hashlib
import streamlit as st
import bcrypt

from database import (
    create_user, get_user, get_all_users, update_user_hash,
    user_exists, seed_default_users,
)


# ── Hashing ────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """
    Hash un mot de passe avec bcrypt (sel aléatoire inclus).
    Retourne une chaîne UTF-8 prête à stocker.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, stored_hash: str, hash_type: str) -> bool:
    """
    Vérifie un mot de passe selon le type de hash stocké.

    - 'bcrypt' : vérification bcrypt standard
    - 'sha256' : vérification SHA256 (anciens comptes, migration transparente)
    """
    if hash_type == "bcrypt":
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    elif hash_type == "sha256":
        return hashlib.sha256(password.encode()).hexdigest() == stored_hash
    return False


# ── Compatibilité weekly_report_job.py ────────────────────────────────────
def load_users() -> dict:
    """
    Retourne tous les utilisateurs sous le format dict attendu par
    weekly_report_job.py :

        { "email": {"company": "...", "secteur": "...", "created_at": "..."} }
    """
    return get_all_users()


# ── Comptes de démonstration ───────────────────────────────────────────────
seed_default_users(hash_password)


# ── Inscription ────────────────────────────────────────────────────────────
def register_user(email: str, password: str, company: str, secteur: str,
                  role: str = "conseiller"):
    """
    Crée un nouveau compte avec un hash bcrypt.

    Returns:
        (True, "message succès") ou (False, "message d'erreur")
    """
    if user_exists(email):
        return False, "Cet email est déjà utilisé."

    pw_hash = hash_password(password)
    try:
        create_user(email, pw_hash, company, secteur, hash_type="bcrypt", role=role)
    except Exception as e:
        return False, f"Erreur lors de la création du compte : {e}"

    return True, "Compte créé avec succès !"


# ── Connexion ──────────────────────────────────────────────────────────────
def login_user(email: str, password: str):
    """
    Vérifie les identifiants.

    Migration transparente : si le compte utilise encore SHA256,
    le mot de passe est re-hashé en bcrypt après vérification réussie.

    Returns:
        (True, user_dict) ou (False, "message d'erreur")
    """
    user = get_user(email)

    if user is None:
        return False, "Email introuvable."

    if not _check_password(password, user["password_hash"], user["hash_type"]):
        return False, "Mot de passe incorrect."

    # Migration transparente SHA256 → bcrypt
    if user["hash_type"] == "sha256":
        new_hash = hash_password(password)
        update_user_hash(email, new_hash, new_type="bcrypt")

    return True, {
        "company":    user["company"],
        "secteur":    user["secteur"],
        "role":       user.get("role", "conseiller"),
        "created_at": user["created_at"],
    }


# ── Page Login / Inscription (Streamlit) ──────────────────────────────────
def show_auth_page():
    st.markdown("""
    <div style='max-width:480px;margin:60px auto;'>
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                    padding:2rem;border-radius:20px;text-align:center;
                    margin-bottom:2rem;color:white;'>
            <h1 style='margin:0;font-size:2.5rem;'>🔮 RetainIQ</h1>
            <p style='margin:8px 0 0 0;opacity:0.9;'>
                Plateforme IA de Prédiction du Churn
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Se connecter", "📝 Créer un compte"])

    with tab1:
        with st.form("login_form"):
            email    = st.text_input("Email",        placeholder="votre@email.com")
            password = st.text_input("Mot de passe", type="password")
            submit   = st.form_submit_button("Se connecter", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Veuillez remplir tous les champs.")
            else:
                ok, result = login_user(email, password)
                if ok:
                    st.session_state.logged_in    = True
                    st.session_state.user_email   = email
                    st.session_state.user_company = result["company"]
                    st.session_state.user_secteur = result["secteur"]
                    st.session_state.user_role    = result.get("role", "conseiller")
                    st.success("Connexion réussie !")
                    st.rerun()
                else:
                    st.error(result)

    with tab2:
        with st.form("register_form"):
            company   = st.text_input("Nom de votre entreprise", placeholder="Ex: Orange, Fitness Park...")
            email2    = st.text_input("Email professionnel",     placeholder="votre@email.com")
            secteur   = st.selectbox("Secteur d'activité", [
                "📱 Télécom", "💪 Salle de Sport",
                "🛍️ E-commerce", "🎓 EdTech", "☁️ SaaS B2B"
            ])
            password2 = st.text_input("Mot de passe",            type="password", key="reg_pwd")
            confirm   = st.text_input("Confirmer le mot de passe", type="password")
            submit2   = st.form_submit_button("Créer mon compte", use_container_width=True)

        if submit2:
            if not all([company, email2, password2, confirm]):
                st.error("Veuillez remplir tous les champs.")
            elif password2 != confirm:
                st.error("Les mots de passe ne correspondent pas.")
            elif len(password2) < 6:
                st.error("Le mot de passe doit contenir au moins 6 caractères.")
            else:
                ok, msg = register_user(email2, password2, company, secteur)
                if ok:
                    st.success(msg + " Vous pouvez maintenant vous connecter.")
                else:
                    st.error(msg)
