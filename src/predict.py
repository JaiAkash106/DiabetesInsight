"""Prediction utilities for the diabetes ML app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


class InputValidationError(ValueError):
    """Raised when user input fails validation."""


class PredictionError(RuntimeError):
    """Raised when prediction execution fails."""


FEATURE_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "Pregnancies": {"min": 0, "max": 20},
    "Glucose": {"min": 0, "max": 250},
    "BloodPressure": {"min": 0, "max": 200},
    "SkinThickness": {"min": 0, "max": 100},
    "Insulin": {"min": 0, "max": 900},
    "BMI": {"min": 10.0, "max": 70.0},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0},
    "GeneticRisk": {"min": 0, "max": 1},
    "Age": {"min": 1, "max": 120},
}


def load_model(model_path: str):
    """Load the trained model pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found at '{path}'.")

    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive for runtime file corruption.
        raise ModelLoadError("Unable to load the trained model.") from exc


def validate_inputs(input_dict: dict) -> dict:
    """Validate numeric ranges and required fields before prediction."""
    sanitized: dict = {}
    missing_fields = [field for field in FEATURE_CONSTRAINTS if field not in input_dict]
    if missing_fields:
        raise InputValidationError(f"Missing input fields: {', '.join(missing_fields)}.")

    for field, limits in FEATURE_CONSTRAINTS.items():
        value = input_dict.get(field)
        if value is None or value == "":
            raise InputValidationError(f"Invalid input value for {field}.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"Invalid input value for {field}.") from exc

        if numeric_value < limits["min"] or numeric_value > limits["max"]:
            raise InputValidationError(
                f"{field} must be between {limits['min']} and {limits['max']}."
            )

        if field in {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "GeneticRisk", "Age"}:
            numeric_value = int(numeric_value)

        sanitized[field] = numeric_value

    return sanitized


def predict_single(model, input_dict: dict) -> Tuple[str, float]:
    """Predict diabetes risk for a single input dictionary."""
    validated_input = validate_inputs(input_dict)
    input_df = pd.DataFrame([validated_input])

    try:
        pred = model.predict(input_df)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba_vals = model.predict_proba(input_df)[0]
            proba = float(proba_vals[1]) if len(proba_vals) == 2 else float(max(proba_vals))
    except InputValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive for runtime inference failures.
        raise PredictionError("Prediction failed. Please try again.") from exc

    label = "Diabetic" if str(pred).lower() in {"1", "yes", "diabetic", "true"} else "Not Diabetic"
    return label, float(proba) if proba is not None else 0.0
