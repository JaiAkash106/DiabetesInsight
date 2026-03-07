"""Pydantic request/response models for API contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class PredictionInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    GeneticRisk: int = Field(ge=0, le=1)
    Age: float


class PredictRequest(BaseModel):
    input: PredictionInput


class PredictResponse(BaseModel):
    prediction_id: int
    label: str
    probability: float
    latency_ms: float


class JobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result: PredictResponse | None = None
    error: str | None = None


class HistoryItem(BaseModel):
    id: int
    input: dict
    prediction_label: str
    probability: float
    latency_ms: float
    created_at: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
