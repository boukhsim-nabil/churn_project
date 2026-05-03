import os

import requests
from dotenv import load_dotenv

load_dotenv()

_BREVO_SMTP_ENDPOINT = "https://api.brevo.com/v3/smtp/email"


def send_campaign_email(to_email: str, subject: str, html_content: str) -> tuple[bool, str]:
    """
    Send an email via the Brevo (ex-Sendinblue) API v3.

    Returns:
        (success: bool, message: str)
    """
    api_key = os.getenv("BREVO_API_KEY")
    from_email = os.getenv("FROM_EMAIL")
    from_name = os.getenv("SENDER_NAME", "RetainIQ")

    if not api_key or api_key.startswith("xkeysib-xxx"):
        return False, (
            "BREVO_API_KEY manquante ou non configurée dans le fichier .env. "
            "Ajoutez : BREVO_API_KEY=xkeysib-votre_cle"
        )

    if not from_email:
        return False, (
            "FROM_EMAIL manquante dans le fichier .env. "
            "Ajoutez : FROM_EMAIL=votre@domaine.com"
        )

    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content,
    }

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            _BREVO_SMTP_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=15,
        )

        if 200 <= response.status_code < 300:
            return True, f"Email envoyé avec succès à {to_email}"

        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text

        return False, f"Brevo a retourné le code HTTP {response.status_code} : {detail}"

    except requests.exceptions.Timeout:
        return False, "Délai d'attente dépassé lors de la connexion à l'API Brevo (timeout 15s)."
    except requests.exceptions.ConnectionError:
        return False, "Impossible de joindre l'API Brevo — vérifiez votre connexion réseau."
    except Exception as exc:
        return False, f"Erreur inattendue lors de l'envoi : {exc}"
