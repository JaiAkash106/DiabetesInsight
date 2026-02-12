"""Preprocessing utilities for the diabetes ML app."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(csv_path)


def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, list[str]]:
    """Create a ColumnTransformer that handles missing values and encoding.

    Returns the fitted preprocessor and the list of feature columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != target_col]

    numeric_features = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, feature_cols
