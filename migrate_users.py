"""
migrate_users.py — Migration one-shot de users.json vers SQLite.

Les mots de passe SHA256 sont conservés tels quels avec hash_type='sha256'.
Ils seront automatiquement convertis en bcrypt à la prochaine connexion.

Usage :
    python migrate_users.py
"""

import json
import os
import sys
from datetime import datetime

# Assure que le dossier courant est dans le path
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, get_connection, user_exists

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def migrate():
    print("=== Migration users.json -> SQLite ===\n")

    # Vérifier que le fichier source existe
    if not os.path.exists(USERS_FILE):
        print(f"[INFO] {USERS_FILE} introuvable — rien à migrer.")
        return

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)

    if not users:
        print("[INFO] users.json est vide — rien à migrer.")
        return

    print(f"[INFO] {len(users)} utilisateur(s) trouvé(s) dans users.json\n")

    # S'assurer que la DB est initialisée
    init_db()

    migrated = 0
    skipped  = 0
    errors   = 0

    for email, info in users.items():
        if user_exists(email):
            print(f"  [SKIP]  {email} — déjà présent dans SQLite")
            skipped += 1
            continue

        password_hash = info.get("password", "")
        company       = info.get("company",   "")
        secteur       = info.get("secteur",   "")
        created_at    = info.get("created_at", datetime.now().isoformat(sep=" ", timespec="seconds"))

        # Normalise created_at au format attendu
        try:
            # Tronque les microsecondes si présentes
            created_at = str(created_at)[:19]
        except Exception:
            created_at = datetime.now().isoformat(sep=" ", timespec="seconds")

        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO users (email, password_hash, hash_type, company, secteur, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (email, password_hash, "sha256", company, secteur, created_at),
                )
            print(f"  [OK]    {email} — migré (hash_type=sha256, migration bcrypt au prochain login)")
            migrated += 1

        except Exception as e:
            print(f"  [ERROR] {email} — {e}")
            errors += 1

    print(f"\n=== Résultat ===")
    print(f"  Migrés   : {migrated}")
    print(f"  Ignorés  : {skipped} (déjà dans SQLite)")
    print(f"  Erreurs  : {errors}")

    if migrated > 0:
        print(f"\n[INFO] La base SQLite est prête : retainiq.db")
        print("[INFO] Les mots de passe seront convertis en bcrypt à la prochaine connexion.")

    if errors == 0 and migrated > 0:
        backup_path = USERS_FILE + ".bak"
        import shutil
        shutil.copy2(USERS_FILE, backup_path)
        print(f"\n[INFO] Sauvegarde de users.json créée : {backup_path}")
        print("[INFO] Vous pouvez supprimer users.json manuellement si tout fonctionne.")


if __name__ == "__main__":
    migrate()
