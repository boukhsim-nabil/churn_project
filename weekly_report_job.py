import os
import tempfile
from datetime import datetime

from auth import load_users
from data_pipeline import load_user_model
from email_reports import generate_pdf_report, send_pdf_via_sendgrid
from database import get_report_recipients


def send_weekly_reports(user_email: str = None):
    sender_email = os.getenv("FROM_EMAIL")
    sender_name = os.getenv("SENDER_NAME", "RetainIQ")

    if not sender_email:
        raise RuntimeError("FROM_EMAIL manquant dans les variables d'environnement.")

    users = load_users()
    # If called with a specific user (manual trigger), process only that one
    targets = {user_email: users.get(user_email, {})} if user_email else users

    for email, info in targets.items():
        company = info.get("company", "Mon Entreprise")
        sector = info.get("secteur", "📱 Télécom")

        model, features, custom_df = load_user_model(email)

        if model is None or custom_df is None:
            print(f"[SKIP] Aucun modèle/données pour {email}")
            continue

        df = custom_df.copy()

        if "ChurnProba" not in df.columns:
            X = df.drop("Churn", axis=1, errors="ignore")
            if features:
                X = X[[c for c in features if c in X.columns]]
            X = X.select_dtypes(include=["number", "bool", "category"])
            if X.empty:
                print(f"[SKIP] DataFrame vide pour {email}")
                continue
            df["ChurnProba"] = model.predict_proba(X)[:, 1]

        if "RiskLevel" not in df.columns:
            df["RiskLevel"] = df["ChurnProba"].apply(
                lambda x: "🔴 Risque Élevé" if x > 0.6 else ("🟡 Risque Modéré" if x > 0.35 else "🟢 Risque Faible")
            )

        recipients = get_report_recipients(email)
        if not recipients:
            print(f"[SKIP] Aucun destinataire configuré pour {email}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(
                tmpdir,
                f"retainiq_weekly_report_{email.replace('@', '_at_').replace('.', '_')}.pdf"
            )

            generate_pdf_report(
                df=df,
                company_name=company,
                sector=sector,
                output_path=pdf_path,
                report_title="Rapport hebdomadaire RetainIQ",
            )

            subject = f"RetainIQ - Rapport hebdomadaire churn - {company}"
            body = (
                f"Bonjour,\n\n"
                f"Voici votre rapport hebdomadaire RetainIQ pour {company}.\n"
                f"Vous trouverez en pièce jointe la liste des clients à risque et les recommandations.\n\n"
                f"Bien cordialement,\n"
                f"RetainIQ"
            )

            for recipient in recipients:
                ok, msg = send_pdf_via_sendgrid(
                    to_email=recipient,
                    subject=subject,
                    body_text=body,
                    pdf_path=pdf_path,
                    from_email=sender_email,
                    from_name=sender_name,
                )
                if ok:
                    print(f"[OK] {msg}")
                else:
                    print(f"[FALLBACK] {msg}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"=== RetainIQ weekly job started at {datetime.now()} ===")
    send_weekly_reports(target)
    print("=== Job terminé ===")
