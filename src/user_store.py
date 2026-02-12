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


def ensure_users_csv(csv_path: str | Path) -> Path:
    """Create users CSV with required columns if it doesn't exist."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=USER_COLUMNS).to_csv(path, index=False)
    return path


def _load_users(csv_path: str | Path) -> pd.DataFrame:
    path = ensure_users_csv(csv_path)
    df = pd.read_csv(path)

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
    df.to_csv(ensure_users_csv(csv_path), index=False)
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
    df.to_csv(ensure_users_csv(csv_path), index=False)
