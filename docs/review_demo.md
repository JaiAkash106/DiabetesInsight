# Review Demonstration Checklist

## 1) Core AI Logic & Backend
- Start API: `python -m uvicorn src.api:app --reload`
- Register + login using frontend or API
- Call `POST /v1/predict`
- Confirm response fields: `prediction_id`, `label`, `probability`, `latency_ms`

## 2) Latency & Async Handling
- Show `src/services/ai_service.py` using `asyncio.to_thread(...)`
- Start background call: `POST /v1/predict/background`
- Poll `GET /v1/predict/jobs/{job_id}` until completed

## 3) Basic Frontend
- Start UI: `python -m streamlit run frontend_app.py`
- Verify login, prediction request, and response display

## 4) Database & Auth Integration
- Confirm DB file exists: `data/app.db`
- Auth endpoints: `/v1/auth/register`, `/v1/auth/login`
- History endpoint: `GET /v1/me/history`
- Verify prediction entries persist across restarts

## 5) Git Progress & Commit History
- Show timeline: `git log --date=short --pretty=format:"%h %ad %s"`
- For Day 10-Day 20 rubric, keep daily modular commits and avoid one large commit.
