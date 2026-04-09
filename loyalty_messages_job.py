# ═══════════════════════════════════════════════════════════════════════════════
# loyalty_messages_job.py — Job mensuel d'envoi des messages de gratitude
# RetainIQ · Exécution automatique le 1er du mois à 10h00
# ═══════════════════════════════════════════════════════════════════════════════

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from database import get_all_users

load_dotenv()

GMAIL_ADDRESS  = os.getenv("GMAIL_ADDRESS", "")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
SENDER_NAME    = os.getenv("SENDER_NAME", "RetainIQ")


# ── CHARGEMENT DES UTILISATEURS ──────────────────────────────────────────────
def _load_users():
    """Charge tous les utilisateurs depuis SQLite."""
    return get_all_users()


# ── CHARGEMENT DU MODÈLE / DONNÉES D'UN UTILISATEUR ─────────────────────────
def _load_user_data(user_email):
    """
    Charge les données nettoyées et le modèle de l'utilisateur.
    Retourne (df, model, features) ou (None, None, None) si absent.
    """
    try:
        import pickle
        import pandas as pd

        safe_email = user_email.replace("@", "_at_").replace(".", "_")
        data_path  = f"data_{safe_email}.csv"
        model_path = f"model_{safe_email}.pkl"

        if not os.path.exists(data_path):
            return None, None, None

        df = pd.read_csv(data_path)

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                saved   = pickle.load(f)
            model    = saved["model"]
            features = saved["features"]
        else:
            model, features = None, None

        return df, model, features

    except Exception as e:
        print(f"  ⚠️ Erreur chargement données {user_email}: {e}")
        return None, None, None


# ── FILTRER LES CHAMPIONS ────────────────────────────────────────────────────
def _filter_champions(df, model, features):
    """
    Calcule ChurnProba si nécessaire, puis filtre les Champions :
    ChurnProba < 0.20 ET tenure >= 12 mois
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Calculer ChurnProba si pas encore fait
    if 'ChurnProba' not in df.columns and model is not None and features is not None:
        try:
            X = df[[f for f in features if f in df.columns]]
            df['ChurnProba'] = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  ⚠️ Erreur calcul ChurnProba: {e}")
            return pd.DataFrame()

    if 'ChurnProba' not in df.columns or 'tenure' not in df.columns:
        return pd.DataFrame()

    champions = df[
        (df['ChurnProba'] < 0.20) &
        (df['tenure']     >= 12)
    ].copy()

    return champions


# ── CONSTRUCTION DU MESSAGE EMAIL ────────────────────────────────────────────
def _build_message(client_idx, tenure, secteur, type_msg):
    """
    Construit le message de gratitude selon le type (anniversaire / mensuel).
    Utilise les templates de loyalty_config.py.
    """
    try:
        from loyalty_config import GRATITUDE_MESSAGES, REWARDS_CATALOG

        messages = GRATITUDE_MESSAGES.get(secteur, GRATITUDE_MESSAGES.get("📱 Télécom", {}))
        catalog  = REWARDS_CATALOG.get(secteur, {})

        annees = int(tenure) // 12
        mois_r = int(tenure) % 12

        if type_msg == "anniversaire":
            # Choisir une récompense de fidélité automatiquement
            fidelite_rewards = catalog.get("fidelite", ["un cadeau exclusif"])
            reward = fidelite_rewards[client_idx % len(fidelite_rewards)]

            msg = messages.get("anniversaire", "Merci pour votre fidélité !").format(
                annees=annees,
                tenure=int(tenure),
                reward=reward
            )
            subject = f"🎂 Joyeux anniversaire de contrat — {annees} an(s) avec nous !"

        else:  # mensuel
            msg = messages.get("mensuel", "Merci pour votre confiance !").format(
                tenure=int(tenure)
            )
            subject = f"💌 Merci pour votre fidélité — RetainIQ"

        return subject, msg

    except Exception as e:
        print(f"  ⚠️ Erreur construction message: {e}")
        return "Message RetainIQ", "Merci pour votre fidélité."


# ── ENVOI EMAIL SIMPLE ────────────────────────────────────────────────────────
def _send_gratitude_email(to_email, subject, body_text):
    """
    Envoie un email de gratitude via Gmail SMTP.
    Retourne True si succès, False sinon.
    """
    if not GMAIL_ADDRESS or not GMAIL_PASSWORD:
        print("  ⚠️ Gmail non configuré — email non envoyé (simulé)")
        return True  # Simule le succès si pas de config email

    try:
        msg            = MIMEMultipart('alternative')
        msg['From']    = f"{SENDER_NAME} <{GMAIL_ADDRESS}>"
        msg['To']      = to_email
        msg['Subject'] = subject

        # Corps texte
        msg.attach(MIMEText(body_text, 'plain', 'utf-8'))

        # Corps HTML
        html_body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#f8fafc;padding:20px;">
            <div style="max-width:580px;margin:0 auto;background:#0A1628;border-radius:12px;overflow:hidden;">
                <div style="padding:24px 30px;text-align:center;">
                    <h1 style="color:#02C39A;margin:0;font-size:24px;">🔮 RetainIQ</h1>
                    <p style="color:#CBD5E1;margin:6px 0 0;font-size:13px;">Message de Fidélité & Gratitude</p>
                </div>
                <div style="padding:24px 30px;background:#0D1B2E;">
                    <div style="background:#1E3A5F;border-left:4px solid #02C39A;
                                padding:16px;border-radius:0 8px 8px 0;margin-bottom:16px;">
                        <pre style="color:#F1F5F9;font-size:13px;white-space:pre-wrap;
                                    font-family:Arial,sans-serif;margin:0;line-height:1.6;">
{body_text}
                        </pre>
                    </div>
                    <p style="color:#64748B;font-size:11px;text-align:center;margin-top:16px;">
                        Message envoyé automatiquement par RetainIQ · {datetime.now().strftime('%d/%m/%Y')}
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

        return True

    except Exception as e:
        print(f"  ❌ Erreur envoi email à {to_email}: {e}")
        return False


# ── JOB PRINCIPAL ─────────────────────────────────────────────────────────────
def send_loyalty_messages():
    """
    Job principal — exécuté le 1er du mois à 10h00 par APScheduler.

    Pour chaque utilisateur de la plateforme :
    1. Charger ses données + modèle
    2. Filtrer les Champions (ChurnProba < 0.20, tenure >= 12)
    3. Pour chaque champion :
       - Si tenure % 12 == 0 → Message d'anniversaire de contrat
       - Sinon               → Message de reconnaissance mensuelle
    4. Logger les résultats
    """
    print(f"\n{'='*60}")
    print(f"  🏆 JOB FIDÉLITÉ — {datetime.now().strftime('%d/%m/%Y à %Hh%M')}")
    print(f"{'='*60}")

    users = _load_users()
    if not users:
        print("  ⚠️ Aucun utilisateur trouvé dans users.json")
        return {"status": "skipped", "reason": "no_users", "total_sent": 0}

    total_sent        = 0
    total_anniversaire = 0
    total_mensuel     = 0
    erreurs           = []

    for user_email, user_data in users.items():
        company_name = user_data.get("company", "Entreprise")
        secteur      = user_data.get("secteur", "📱 Télécom")

        print(f"\n  👤 Traitement : {company_name} ({user_email})")

        # Charger les données
        df, model, features = _load_user_data(user_email)
        if df is None:
            print(f"  ⏭️  Pas de données — SKIP")
            continue

        # Filtrer les Champions
        champions = _filter_champions(df, model, features)
        if champions.empty:
            print(f"  ℹ️  0 champion détecté — SKIP")
            continue

        print(f"  🌟 {len(champions)} champions détectés")

        # Traiter chaque champion
        n_sent_user = 0
        for i, (_, row) in enumerate(champions.iterrows()):
            tenure = int(row.get('tenure', 0))

            # ── Logique du modulo ─────────────────────────────────
            # tenure % 12 == 0 → anniversaire de contrat
            # tenure % 12 != 0 → message de reconnaissance mensuelle
            if tenure % 12 == 0 and tenure > 0:
                type_msg = "anniversaire"
                total_anniversaire += 1
            else:
                type_msg = "mensuel"
                total_mensuel += 1

            # Construire le message
            subject, body = _build_message(i, tenure, secteur, type_msg)

            # Envoyer à l'email de l'entreprise (en prod : email du client)
            # Pour la démo, on envoie à l'email de l'utilisateur inscrit
            success = _send_gratitude_email(user_email, subject, body)

            if success:
                n_sent_user += 1
                total_sent  += 1
                icon = "🎂" if type_msg == "anniversaire" else "💌"
                print(f"    {icon} Client #{i+1} — tenure={tenure}m → {type_msg} — ✅ envoyé")
            else:
                erreurs.append(f"{user_email} — Client #{i+1}")

        print(f"  ✅ {n_sent_user}/{len(champions)} messages envoyés pour {company_name}")

    # ── Rapport final ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📊 RÉSUMÉ DU JOB FIDÉLITÉ")
    print(f"  Total messages envoyés : {total_sent}")
    print(f"  Anniversaires          : {total_anniversaire}")
    print(f"  Reconnaissances        : {total_mensuel}")
    print(f"  Erreurs                : {len(erreurs)}")
    print(f"{'='*60}\n")

    return {
        "status":              "success" if not erreurs else "partial",
        "total_sent":          total_sent,
        "anniversaires":       total_anniversaire,
        "mensuels":            total_mensuel,
        "erreurs":             erreurs,
        "executed_at":         datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


# ── TEST STANDALONE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lancement du job de fidélité en mode test...")
    result = send_loyalty_messages()
    print(f"\nRésultat : {result}")