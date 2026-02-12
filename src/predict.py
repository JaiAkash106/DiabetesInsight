"""Prediction utilities for the diabetes ML app."""

from __future__ import annotations

from typing import Tuple

import joblib
import pandas as pd


def load_model(model_path: str):
    """Load the trained model pipeline."""
    return joblib.load(model_path)


def predict_single(model, input_dict: dict) -> Tuple[str, float]:
    """Predict diabetes risk for a single input dictionary.

    Returns label and probability of the positive class if available.
    """
    input_df = pd.DataFrame([input_dict])
    pred = model.predict(input_df)[0]

    # Try to get probability for the positive class (assumes label 1 or 'Yes')
    proba = None
    if hasattr(model, "predict_proba"):
        proba_vals = model.predict_proba(input_df)[0]
        if len(proba_vals) == 2:
            # Take probability of the positive class (assumed index 1)
            proba = float(proba_vals[1])
        else:
            proba = float(max(proba_vals))

    label = "Diabetic" if str(pred).lower() in {"1", "yes", "diabetic", "true"} else "Not Diabetic"
    return label, float(proba) if proba is not None else 0.0
