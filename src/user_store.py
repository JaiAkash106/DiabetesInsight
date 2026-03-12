"""CSV-backed user storage for Streamlit login and prediction history."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

USER_COLUMNS = [
    "name",
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "genetic_risk",
    "age",
    "prediction",
    "probability",
]


class UserStoreError(RuntimeError):
    """Raised when the CSV-backed store cannot be accessed safely."""


def ensure_users_csv(csv_path: str | Path) -> Path:
    """Create users CSV with required columns if it doesn't exist."""
    path = Path(csv_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            pd.DataFrame(columns=USER_COLUMNS).to_csv(path, index=False)
    except Exception as exc:  # pragma: no cover - filesystem safety.
        raise UserStoreError("Unable to initialize user storage.") from exc
    return path


def _load_users(csv_path: str | Path) -> pd.DataFrame:
    path = ensure_users_csv(csv_path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - filesystem safety.
        raise UserStoreError("Unable to read saved user data.") from exc

    # Ensure consistent schema even if file was modified manually.
    for col in USER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[USER_COLUMNS]


def user_exists(name: str, csv_path: str | Path) -> bool:
    """Return True when a user row already exists for this name."""
    df = _load_users(csv_path)
    return (df["name"] == name).any()


def create_user(name: str, csv_path: str | Path) -> bool:
    """Insert a new user row; return False if name already exists."""
    df = _load_users(csv_path)
    if (df["name"] == name).any():
        return False

    row = {col: pd.NA for col in USER_COLUMNS}
    row["name"] = name
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    try:
        df.to_csv(ensure_users_csv(csv_path), index=False)
    except Exception as exc:  # pragma: no cover - filesystem safety.
        raise UserStoreError("Unable to save user profile.") from exc
    return True


def upsert_user_prediction(
    name: str,
    input_values: Dict[str, float | int],
    prediction: str,
    probability: float,
    csv_path: str | Path,
) -> None:
    """Update existing user row; insert user if row is missing."""
    df = _load_users(csv_path)
    mask = df["name"] == name

    if not mask.any():
        row = {col: pd.NA for col in USER_COLUMNS}
        row["name"] = name
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        mask = df["name"] == name

    for key, value in input_values.items():
        if key in df.columns:
            df.loc[mask, key] = value

    df.loc[mask, "prediction"] = prediction
    df.loc[mask, "probability"] = probability
    try:
        df.to_csv(ensure_users_csv(csv_path), index=False)
    except Exception as exc:  # pragma: no cover - filesystem safety.
        raise UserStoreError("Unable to save prediction history.") from exc
