"""Runtime configuration for backend and frontend clients."""

from __future__ import annotations

import os

DATA_PATH = os.getenv("DATA_PATH", "data/diabetes.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
TARGET_COL = os.getenv("TARGET_COL", "Outcome")
DB_PATH = os.getenv("DB_PATH", "data/app.db")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")
