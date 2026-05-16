"""
scheduler.py — Singleton APScheduler pour les rapports hebdomadaires RetainIQ.

Le scheduler est initialisé une seule fois au niveau module.
Streamlit re-exécute le script à chaque interaction, mais le module
n'est chargé qu'une seule fois par processus Python, ce qui garantit
une seule instance du scheduler.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loyalty_messages_job import send_loyalty_messages



# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retainiq.scheduler")

# ── État global du scheduler ───────────────────────────────────────────────
_scheduler: Optional[BackgroundScheduler] = None
_lock = Lock()

# Historique des exécutions (max 50 entrées, conservé en mémoire)
run_history: list[dict] = []

# Prochaine exécution planifiée (mise à jour après chaque run)
next_run_time: Optional[datetime] = None


# ── Wrapper du job ─────────────────────────────────────────────────────────
def _run_weekly_job(user_email: str = None):
    """Appelé par APScheduler ou trigger_now(). Exécute send_weekly_reports() et journalise."""
    from weekly_report_job import send_weekly_reports  # import tardif pour éviter les cycles
    from database import get_all_users

    started_at = datetime.now()
    logger.info(f"[Scheduler] Démarrage du rapport hebdomadaire à {started_at:%Y-%m-%d %H:%M:%S}")

    try:
        if user_email:
            send_weekly_reports(user_email)
            detail = f"Rapport envoyé pour {user_email}."
        else:
            # Cron automatique : envoie pour tous les utilisateurs
            for email in get_all_users():
                send_weekly_reports(email)
            detail = "Rapports envoyés à tous les utilisateurs."
        status = "✅ Succès"
    except Exception as exc:
        status = "❌ Erreur"
        detail = str(exc)
        logger.exception("[Scheduler] Erreur lors de l'envoi des rapports.")

    finished_at = datetime.now()
    duration = (finished_at - started_at).seconds

    entry = {
        "date": started_at.strftime("%d/%m/%Y %H:%M"),
        "status": status,
        "detail": detail,
        "duration_s": duration,
    }

    run_history.insert(0, entry)
    if len(run_history) > 50:
        run_history.pop()

    logger.info(f"[Scheduler] Terminé — {status} en {duration}s")


# ── Démarrage du scheduler ─────────────────────────────────────────────────
def start_scheduler(
    day_of_week: str = "mon",
    hour: int = 8,
    minute: int = 0,
) -> BackgroundScheduler:
    """
    Démarre le BackgroundScheduler si ce n'est pas déjà fait.
    Planifie le rapport hebdomadaire selon le cron fourni.

    Args:
        day_of_week: Jour de la semaine (ex: 'mon', 'tue', ... 'sun')
        hour:        Heure d'envoi (0-23)
        minute:      Minute d'envoi (0-59)

    Returns:
        L'instance du scheduler (déjà démarrée).
    """
    global _scheduler

    with _lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        _scheduler = BackgroundScheduler(timezone="Europe/Paris")

        _scheduler.add_job(
            func=_run_weekly_job,
            trigger=CronTrigger(
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                timezone="Europe/Paris",
            ),
            id="weekly_report",
            name="Rapport hebdomadaire RetainIQ",
            replace_existing=True,
            misfire_grace_time=3600,  # tolérance 1h si Streamlit était éteint
        )
        # Job fidélité — le 1er de chaque mois à 10h00
        _scheduler.add_job(
            func=send_loyalty_messages,
            trigger=CronTrigger(day=1, hour=10, minute=0),
            id="loyalty_messages",
            misfire_grace_time=3600
        )


        _scheduler.start()
        logger.info(
            f"[Scheduler] Démarré — rapport planifié chaque {day_of_week} à {hour:02d}:{minute:02d}"
        )



    _refresh_next_run()
    return _scheduler


def stop_scheduler():
    """Arrête le scheduler proprement (utile pour les tests)."""
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Arrêté.")
        _scheduler = None


def get_status() -> dict:
    """
    Retourne l'état courant du scheduler sous forme de dictionnaire.

    Keys:
        running (bool)          — Le scheduler tourne-t-il ?
        next_run (str | None)   — Prochaine exécution formatée
        job_count (int)         — Nombre de jobs enregistrés
        history (list[dict])    — 10 dernières exécutions
    """
    _refresh_next_run()
    running = _scheduler is not None and _scheduler.running

    return {
        "running": running,
        "next_run": next_run_time.strftime("%d/%m/%Y à %H:%M") if next_run_time else "—",
        "job_count": len(_scheduler.get_jobs()) if running else 0,
        "history": run_history[:10],
    }


def trigger_now(user_email: str = None):
    """Lance le job immédiatement (utile depuis le dashboard pour un test)."""
    _run_weekly_job(user_email)


def update_schedule(day_of_week: str, hour: int, minute: int):
    """
    Met à jour la planification sans redémarrer le scheduler.

    Args:
        day_of_week: ex 'mon', 'fri'
        hour:        0-23
        minute:      0-59
    """
    if _scheduler is None or not _scheduler.running:
        start_scheduler(day_of_week=day_of_week, hour=hour, minute=minute)
        return

    _scheduler.reschedule_job(
        job_id="weekly_report",
        trigger=CronTrigger(
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone="Europe/Paris",
        ),
    )
    logger.info(f"[Scheduler] Planification mise à jour → {day_of_week} à {hour:02d}:{minute:02d}")
    _refresh_next_run()


# ── Utilitaires internes ───────────────────────────────────────────────────
def _refresh_next_run():
    """Met à jour la variable globale next_run_time."""
    global next_run_time
    if _scheduler and _scheduler.running:
        job = _scheduler.get_job("weekly_report")
        if job and job.next_run_time:
            next_run_time = job.next_run_time.replace(tzinfo=None)
        else:
            next_run_time = None
    else:
        next_run_time = None
