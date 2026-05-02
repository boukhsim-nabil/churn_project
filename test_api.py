import os
import google.generativeai as genai
from dotenv import load_dotenv

# Charge la clé depuis le fichier .env
load_dotenv()
cle_api = os.getenv("GEMINI_API_KEY")

print(f"Clé chargée : {cle_api[:10]}... (cachée pour sécurité)")

genai.configure(api_key=cle_api)

print("\nVoici les modèles EXACTS autorisés pour ta clé :")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"👉 {m.name}")
except Exception as e:
    print(f"Erreur d'authentification : {e}")