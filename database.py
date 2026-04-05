"""
database.py — Couche SQLite pour RetainIQ.

Remplace users.json par une base de données SQLite légère.
Appelé par auth.py. Ne dépend d'aucun autre module du projet.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "retainiq.db")


# ── Connexion ──────────────────────────────────────────────────────────────
@contextmanager
def get_connection():
    """Context manager : ouvre une connexion, commit ou rollback, puis ferme."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # accès aux colonnes par nom
    conn.execute("PRAGMA journal_mode=WAL")   # meilleure concurrence en lecture
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation du schéma ───────────────────────────────────────────────
def init_db():
    """
    Crée la table users si elle n'existe pas.
    Sûr à appeler plusieurs fois (CREATE TABLE IF NOT EXISTS).
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email          TEXT PRIMARY KEY,
                password_hash  TEXT NOT NULL,
                hash_type      TEXT NOT NULL DEFAULT 'bcrypt',
                company        TEXT NOT NULL DEFAULT '',
                secteur        TEXT NOT NULL DEFAULT '',
                created_at     TEXT NOT NULL
            )
        """)


# ── Opérations CRUD ────────────────────────────────────────────────────────
def get_user(email: str) -> dict | None:
    """
    Retourne l'utilisateur correspondant à l'email, ou None s'il n'existe pas.

    Returns:
        dict avec clés : email, password_hash, hash_type, company, secteur, created_at
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def create_user(
    email: str,
    password_hash: str,
    company: str,
    secteur: str,
    hash_type: str = "bcrypt",
) -> None:
    """
    Insère un nouvel utilisateur.

    Raises:
        ValueError: si l'email est déjà utilisé.
    """
    if get_user(email) is not None:
        raise ValueError(f"L'email '{email}' est déjà utilisé.")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (email, password_hash, hash_type, company, secteur, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (email, password_hash, hash_type, company, secteur,
             datetime.now().isoformat(sep=" ", timespec="seconds")),
        )


def update_user_hash(email: str, new_hash: str, new_type: str = "bcrypt") -> None:
    """
    Met à jour le hash du mot de passe d'un utilisateur.
    Utilisé pour la migration transparente SHA256 → bcrypt.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, hash_type = ? WHERE email = ?",
            (new_hash, new_type, email),
        )


def get_all_users() -> dict:
    """
    Retourne tous les utilisateurs sous la forme attendue par weekly_report_job.py :

        {
            "email@example.com": {
                "company":    "Nom Entreprise",
                "secteur":    "📱 Télécom",
                "created_at": "2025-01-15 10:30:00",
            },
            ...
        }
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()

    return {
        row["email"]: {
            "company":    row["company"],
            "secteur":    row["secteur"],
            "created_at": row["created_at"],
        }
        for row in rows
    }


def user_exists(email: str) -> bool:
    """Retourne True si l'email est déjà enregistré."""
    return get_user(email) is not None


# ── Bootstrap automatique ──────────────────────────────────────────────────
# La base est initialisée dès l'import du module (idempotent)
init_db()
