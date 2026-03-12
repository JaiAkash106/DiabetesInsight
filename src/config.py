"""Runtime configuration for backend and frontend clients."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str) -> Path:
    """Resolve env-driven paths relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_PATH = str(resolve_path(os.getenv("DATA_PATH", "data/diabetes.csv")))
MODEL_PATH = str(resolve_path(os.getenv("MODEL_PATH", "models/best_model.joblib")))
TARGET_COL = os.getenv("TARGET_COL", "Outcome")
DB_PATH = str(resolve_path(os.getenv("DB_PATH", "data/app.db")))
USERS_CSV_PATH = str(resolve_path(os.getenv("USERS_CSV_PATH", "data/users.csv")))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")
