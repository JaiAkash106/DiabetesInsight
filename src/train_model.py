"""Train and select the best model for the diabetes ML app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, load_dataset


def train_and_select_model(
    csv_path: str,
    target_col: str = "Outcome",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train Logistic Regression and Random Forest, return the best model.

    Returns the best fitted pipeline and a metrics dictionary.
    """
    df = load_dataset(csv_path)
    preprocessor, feature_cols = build_preprocessor(df, target_col)

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
    }

    scores: Dict[str, float] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        scores[name] = accuracy_score(y_test, preds)
        fitted_pipelines[name] = pipeline

    # Select the best model by accuracy
    best_name = max(scores, key=scores.get)
    best_pipeline = fitted_pipelines[best_name]

    metrics = {"best_model": best_name, "accuracy": scores[best_name]}
    return best_pipeline, metrics


def save_model(model: Pipeline, output_path: str) -> None:
    """Persist the trained model pipeline to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def train_and_save(
    csv_path: str,
    target_col: str = "Outcome",
    model_path: str = "models/best_model.joblib",
) -> Dict[str, float]:
    """Train, select, and save the best model."""
    best_model, metrics = train_and_select_model(csv_path, target_col=target_col)
    save_model(best_model, model_path)
    return metrics


if __name__ == "__main__":
    # Allow CLI training: python -m src.train_model
    metrics = train_and_save("data/diabetes.csv", target_col="Outcome")
    print(f"Trained model saved. Best: {metrics['best_model']}, accuracy: {metrics['accuracy']:.4f}")
