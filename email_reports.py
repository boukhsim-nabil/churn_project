import os
import base64
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

import requests


def _safe(v):
    if pd.isna(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _risk_label(score: float) -> str:
    if score > 0.6:
        return "Risque élevé"
    if score > 0.35:
        return "Risque modéré"
    return "Risque faible"


def prepare_scored_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has ChurnProba and RiskLevel columns.
    If ChurnProba already exists, keep it.
    """
    out = df.copy()

    if "ChurnProba" not in out.columns:
        raise ValueError("Le dataframe doit contenir la colonne 'ChurnProba'.")

    if "RiskLevel" not in out.columns:
        out["RiskLevel"] = out["ChurnProba"].apply(
            lambda x: "🔴 Risque Élevé" if x > 0.6 else ("🟡 Risque Modéré" if x > 0.35 else "🟢 Risque Faible")
        )

    return out


def generate_pdf_report(
    df: pd.DataFrame,
    company_name: str,
    sector: str,
    output_path: str,
    report_title: str = "Rapport hebdomadaire churn",
) -> str:
    """
    Generate a professional PDF report for churn monitoring.
    """
    df = prepare_scored_df(df)

    total_clients = len(df)
    high_risk = int((df["ChurnProba"] > 0.6).sum())
    medium_risk = int(((df["ChurnProba"] > 0.35) & (df["ChurnProba"] <= 0.6)).sum())
    low_risk = int((df["ChurnProba"] <= 0.35).sum())
    churn_rate = float(df["Churn"].mean() * 100) if "Churn" in df.columns else 0.0

    top_risk = df.sort_values("ChurnProba", ascending=False).head(10).copy()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleWhite",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#FFFFFF"),
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyGray",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallGray",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#64748B"),
        )
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.2 * cm,
    )

    story = []

    # Header block
    story.append(
        Paragraph(
            f"<para align='center'><b>{report_title}</b></para>",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"<para align='center'>Entreprise : <b>{company_name}</b> | Secteur : <b>{sector}</b> | "
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</para>",
            styles["BodyGray"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # KPI table
    kpi_data = [
        ["Métrique", "Valeur"],
        ["Total clients", f"{total_clients:,}".replace(",", " ")],
        ["Taux de churn réel", f"{churn_rate:.1f}%"],
        ["Clients à risque élevé", str(high_risk)],
        ["Clients à risque modéré", str(medium_risk)],
        ["Clients à risque faible", str(low_risk)],
    ]

    kpi_table = Table(kpi_data, colWidths=[8 * cm, 5.5 * cm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Top 10 clients les plus à risque</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))

    # Client table
    table_headers = ["Client", "Ancienneté", "Charges", "Total", "Score", "Niveau"]
    table_rows = [table_headers]

    for idx, row in top_risk.reset_index(drop=True).iterrows():
        client_label = f"Client {idx + 1}"
        tenure = _safe(row["tenure"]) if "tenure" in row else "-"
        monthly = _safe(row["MonthlyCharges"]) if "MonthlyCharges" in row else "-"
        total = _safe(row["TotalCharges"]) if "TotalCharges" in row else "-"
        score = f"{float(row['ChurnProba']) * 100:.1f}%"
        level = _risk_label(float(row["ChurnProba"]))
        table_rows.append([client_label, str(tenure), str(monthly), str(total), score, level])

    client_table = Table(table_rows, colWidths=[2.2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm, 3.2 * cm])
    client_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(client_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Actions recommandées</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("• Contacter en priorité les clients au-dessus de 60% de risque.", styles["BodyGray"]))
    story.append(Paragraph("• Proposer une offre de rétention personnalisée.", styles["BodyGray"]))
    story.append(Paragraph("• Vérifier les contrats mensuels et les charges élevées.", styles["BodyGray"]))
    story.append(Paragraph("• Suivre l'évolution semaine après semaine.", styles["BodyGray"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Ce document a été généré automatiquement par RetainIQ.",
            styles["SmallGray"],
        )
    )

    doc.build(story)
    return output_path


def _save_pdf_locally(pdf_path: str, to_email: str) -> str:
    """
    Fallback: sauvegarde le PDF localement si SendGrid échoue.
    Retourne le chemin où il a été sauvegardé.
    """
    archive_dir = "reports_archive"
    os.makedirs(archive_dir, exist_ok=True)

    safe_email = to_email.replace("@", "_at_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"report_{safe_email}_{timestamp}.pdf"
    dest_path = os.path.join(archive_dir, dest_name)

    shutil.copy2(pdf_path, dest_path)
    return dest_path


def send_pdf_via_sendgrid(
    to_email: str,
    subject: str,
    body_text: str,
    pdf_path: str,
    from_email: str,
    from_name: str = "RetainIQ",
) -> tuple[bool, str]:
    """
    Envoie le rapport via SendGrid.
    Si SendGrid échoue ou n'est pas configuré, fallback = sauvegarde locale.

    Returns:
        (success: bool, message: str)
    """
    api_key = os.getenv("BREVO_API_KEY")

    # Fallback 1 : clé manquante ou placeholder
    if not api_key or api_key.startswith("xkeysib-xxx"):
        local_path = _save_pdf_locally(pdf_path, to_email)
        msg = f"[FALLBACK] Brevo non configure - PDF sauvegarde localement: {local_path}"
        try:
            print(msg)
        except:
            pass
        return False, msg

    with open(pdf_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode()

    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": body_text,
        "attachment": [{"content": encoded_file, "name": Path(pdf_path).name}],
    }

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            json=payload,
            headers=headers,
            timeout=15,
        )

        if 200 <= response.status_code < 300:
            return True, f"Email envoyé avec succès à {to_email}"

        # Fallback 2 : Brevo a retourné une erreur HTTP
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text
        local_path = _save_pdf_locally(pdf_path, to_email)
        msg = f"[FALLBACK] Brevo a retourné le code HTTP {response.status_code} : {detail} - PDF sauvegardé: {local_path}"
        try:
            print(msg)
        except:
            pass
        return False, msg

    except requests.exceptions.Timeout:
        local_path = _save_pdf_locally(pdf_path, to_email)
        msg = f"[FALLBACK] Délai d'attente dépassé lors de la connexion à l'API Brevo (timeout 15s) - PDF sauvegardé: {local_path}"
        try:
            print(msg)
        except:
            pass
        return False, msg
    except requests.exceptions.ConnectionError:
        local_path = _save_pdf_locally(pdf_path, to_email)
        msg = f"[FALLBACK] Impossible de joindre l'API Brevo — vérifiez votre connexion réseau - PDF sauvegardé: {local_path}"
        try:
            print(msg)
        except:
            pass
        return False, msg
    except Exception as exc:
        # Fallback 3 : erreur inattendue
        local_path = _save_pdf_locally(pdf_path, to_email)
        msg = f"[FALLBACK] Erreur inattendue lors de l'envoi : {exc} - PDF sauvegardé: {local_path}"
        try:
            print(msg)
        except:
            pass
        return False, msg
