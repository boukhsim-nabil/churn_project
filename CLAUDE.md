# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RetainIQ** is an AI-powered churn prediction platform built with Streamlit. It enables companies to predict customer churn, understand risk factors via SHAP explainability, and receive automated weekly email reports. The platform supports 5 business sectors (Telecom, Fitness, E-commerce, EdTech, SaaS B2B) with sector-specific column detection and XGBoost models.

## Stack

- **Frontend/Framework**: Streamlit (1.35+)
- **ML**: XGBoost (2.0+), scikit-learn, Pandas, NumPy
- **Database**: SQLite (retainiq.db) with bcrypt for password hashing
- **Scheduling**: APScheduler (background tasks)
- **Explainability**: SHAP
- **Reporting**: ReportLab (PDF), SendGrid (email)
- **Auth**: bcrypt with SQLite backend (replaces older JSON auth)

## High-Level Architecture

```
churn_prediction_dashboard.py (Streamlit entry point)
├── auth.py → database.py (SQLite user management)
├── data_pipeline.py (CSV import, column detection, XGBoost training)
├── shap_explainer.py (SHAP Tree Explainer + caching)
├── email_reports.py (PDF generation + SendGrid)
└── scheduler.py (APScheduler singleton)
    └── weekly_report_job.py (background email task)
```

**Key data flow:**
1. User logs in → `auth.py` verifies bcrypt hash against SQLite
2. User imports CSV → `data_pipeline.py` auto-detects columns by sector
3. Model trains → XGBoost model saved as `model_[email].pkl`
4. Predictions shown → SHAP explanations via `shap_explainer.py`
5. Weekly job runs (Monday 8am) → `weekly_report_job.py` sends emails via SendGrid

## Critical Design Patterns

### Database & Auth
- **SQLite schema**: `users` table with `email` (PK), `password_hash`, `hash_type`, `company`, `secteur`, `created_at`
- **Auth migration**: Old SHA256 hashes are auto-rehashed to bcrypt on first successful login
- **Sector-specific columns**: `SECTEUR_COLUMNS` dict in `data_pipeline.py` defines required columns per sector

### Streamlit Session State
- `logged_in` (bool): Authentication guard
- `_scheduler_started` (bool): APScheduler singleton init flag
- `user_company`, `user_secteur`: Loaded from SQLite after login
- Models are cached in session state to avoid re-training on re-runs

### Scheduler Pattern
- APScheduler is a module-level singleton in `scheduler.py` (initialized once per process)
- Streamlit re-runs the app on every interaction, but the scheduler persists
- Weekly job runs Monday 8:00 AM, triggered by `_run_weekly_job()`
- History and next run time are tracked in module variables

### Column Detection
- `detect_columns(df, secteur)` in `data_pipeline.py` auto-finds target column and required features
- Falls back to manual mapping if auto-detection fails
- Each sector has specific required columns and target column hints

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (development)
streamlit run churn_prediction_dashboard.py

# Initialize database (called on first app run)
# Already handled by auth.py, but can manually call:
python -c "from database import init_db; init_db()"

# Check database schema
sqlite3 retainiq.db ".schema users"

# View scheduler status (in Streamlit sidebar, "Scheduler Status" section)
# Logs saved to retainiq.log (if configured)
```

## Important Files & Responsibilities

| File | Purpose |
|------|---------|
| `churn_prediction_dashboard.py` | Main Streamlit app, page routing, session management |
| `auth.py` | Login/signup, bcrypt verification, password migration |
| `database.py` | SQLite operations, no other module dependencies |
| `data_pipeline.py` | CSV import, column detection, XGBoost training |
| `shap_explainer.py` | SHAP Tree Explainer, cached explanation generation |
| `email_reports.py` | PDF generation (ReportLab), SendGrid API calls |
| `scheduler.py` | APScheduler singleton, weekly job scheduling |
| `weekly_report_job.py` | Background task: load models, generate PDFs, send emails |

## Testing & Debugging Tips

- **Check auth**: SQLite query `SELECT email, company, secteur FROM users;` to see registered users
- **Verify scheduler**: Check `scheduler.run_history` and `scheduler.next_run_time` in Streamlit sidebar
- **Model files**: Models saved as `model_[email].pkl` in working directory, load with `pickle.load()`
- **SHAP caching**: If explainer is slow, check `shap_cache` in session state
- **SendGrid**: Set `SENDGRID_API_KEY` in `.env` (loaded by `python-dotenv`)

## Known Limitations & TODOs

- Weekly scheduler runs in UTC (verify timezone requirements)
- No admin UI for user management (direct SQLite queries needed)
- SHAP explainer caches entire model in memory (can be large for big datasets)
- No A/B testing framework (all users get same model)

## Environment Variables

Create `.env` with:
```
SENDGRID_API_KEY=your_key_here
```

Loaded by `python-dotenv` in `email_reports.py` and `weekly_report_job.py`.
