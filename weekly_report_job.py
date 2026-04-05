import os
import json
import tempfile
from datetime import datetime

import pandas as pd

from auth import load_users
from data_pipeline import load_user_model
from email_reports import generate_pdf_report, send_pdf_via_sendgrid


def send_weekly_reports():
    users = load_users()
    if not users:
        print("Aucun utilisateur trouvé.")
        return

    sender_email = os.getenv("SENDER_EMAIL")
    sender_name = os.getenv("SENDER_NAME", "RetainIQ")

    if not sender_email:
        raise RuntimeError("SENDER_EMAIL manquant dans les variables d'environnement.")

    for email, info in users.items():
        company = info.get("company", "Mon Entreprise")
        sector = info.get("secteur", "📱 Télécom")

        model, features, custom_df = load_user_model(email)

        if model is None or custom_df is None:
            print(f"[SKIP] Aucun modèle/données pour {email}")
            continue

        df = custom_df.copy()

        if "ChurnProba" not in df.columns:
            X = df.drop("Churn", axis=1, errors="ignore")
            if X.empty:
                print(f"[SKIP] DataFrame vide pour {email}")
                continue
            df["ChurnProba"] = model.predict_proba(X)[:, 1]

        if "RiskLevel" not in df.columns:
            df["RiskLevel"] = df["ChurnProba"].apply(
                lambda x: "🔴 Risque Élevé" if x > 0.6 else ("🟡 Risque Modéré" if x > 0.35 else "🟢 Risque Faible")
            )

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

            ok, message = send_pdf_via_sendgrid(
                to_email=email,
                subject=subject,
                body_text=body,
                pdf_path=pdf_path,
                from_email=sender_email,
                from_name=sender_name,
            )

            if ok:
                print(f"[OK] {message}")
            else:
                print(f"[FALLBACK] {message}")


if __name__ == "__main__":
    print(f"=== RetainIQ weekly job started at {datetime.now()} ===")
    send_weekly_reports()
    print("=== Job terminé ===")
