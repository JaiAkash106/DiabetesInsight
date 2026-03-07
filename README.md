# Diabetes AI

Modular diabetes-risk project with:
- ML training/inference pipeline
- FastAPI backend with auth + prediction history
- Async and background AI processing
- Streamlit frontend draft that consumes backend APIs

## Project Structure

- `src/train_model.py`: train and save model
- `src/predict.py`: inference helpers
- `src/services/ai_service.py`: async AI processing wrapper
- `src/db.py`: SQLite schema and data access
- `src/auth.py`: password hashing and session token helpers
- `src/api.py`: backend API endpoints
- `frontend_app.py`: Streamlit frontend client for backend APIs
- `app.py`: original standalone Streamlit app

## Install

```bash
python -m pip install -r requirements.txt
```

## Train Model

```bash
python -m src.train_model
```

## Run Backend API

```bash
python -m uvicorn src.api:app --reload
```

## Run Frontend Draft

```bash
python -m streamlit run frontend_app.py
```

## Review Criteria Demo Guide

### 1) Core AI Logic & Backend

- Use `POST /v1/predict` with auth token.
- Flow: request payload -> `src/api.py` -> `src/services/ai_service.py` -> model in `src/predict.py` -> response.

### 2) Latency & Async Handling

- `predict_async()` uses `asyncio.to_thread(...)` in `src/services/ai_service.py`.
- Background job flow:
  - `POST /v1/predict/background`
  - `GET /v1/predict/jobs/{job_id}`

### 3) Basic Frontend

- `frontend_app.py` supports:
  - register/login
  - sync or background prediction requests
  - history display from backend

### 4) Database & Auth Integration

- SQLite database: `data/app.db`
- Tables: `users`, `sessions`, `predictions`
- Endpoints:
  - `POST /v1/auth/register`
  - `POST /v1/auth/login`
  - `GET /v1/me/history`

### 5) Git Progress & Commit History

Use this command to show timeline:

```bash
git log --date=short --pretty=format:"%h %ad %s"
```

For day-by-day submission proof (Day 10 to Day 20), maintain daily modular commits instead of one large commit.
