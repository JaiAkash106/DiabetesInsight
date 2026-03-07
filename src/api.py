"""FastAPI backend exposing auth, prediction, and history endpoints."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .db import (
    authenticate_user,
    create_session,
    create_user,
    get_prediction_history,
    init_db,
    resolve_session,
    store_prediction,
)
from .schemas import (
    AuthResponse,
    HealthResponse,
    HistoryResponse,
    JobQueuedResponse,
    JobStatusResponse,
    LoginRequest,
    PredictRequest,
    PredictResponse,
    RegisterRequest,
)
from .services.ai_service import predict_async

app = FastAPI(title="Diabetes AI Backend", version="1.0.0")

JOBS: dict[str, dict[str, Any]] = {}


def _as_dict(model_obj: Any) -> dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


@app.on_event("startup")
async def startup_event() -> None:
    init_db()


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization header")

    parts = authorization.split(" ", maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header")

    return parts[1].strip()


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _extract_bearer_token(authorization)
    user = resolve_session(token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")
    return user


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/v1/auth/register", status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest) -> dict[str, str]:
    created = create_user(payload.username.strip(), payload.password)
    if not created:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    return {"message": "User created"}


@app.post("/v1/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest) -> AuthResponse:
    user_id = authenticate_user(payload.username.strip(), payload.password)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_session(user_id)
    return AuthResponse(access_token=token)


async def _predict_and_store(user_id: int, input_payload: dict[str, Any]) -> PredictResponse:
    result = await predict_async(input_payload)
    prediction_id = store_prediction(
        user_id=user_id,
        input_payload=input_payload,
        prediction_label=result["label"],
        probability=result["probability"],
        latency_ms=result["latency_ms"],
    )

    return PredictResponse(
        prediction_id=prediction_id,
        label=result["label"],
        probability=result["probability"],
        latency_ms=result["latency_ms"],
    )


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, user: dict[str, Any] = Depends(current_user)) -> PredictResponse:
    return await _predict_and_store(user_id=user["user_id"], input_payload=_as_dict(payload.input))


async def _run_background_job(job_id: str, user_id: int, input_payload: dict[str, Any]) -> None:
    JOBS[job_id]["status"] = "running"
    try:
        result = await _predict_and_store(user_id=user_id, input_payload=input_payload)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = _as_dict(result)
    except Exception as exc:  # pragma: no cover
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)


@app.post("/v1/predict/background", response_model=JobQueuedResponse)
async def predict_background(payload: PredictRequest, user: dict[str, Any] = Depends(current_user)) -> JobQueuedResponse:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "result": None, "error": None}
    asyncio.create_task(_run_background_job(job_id, user["user_id"], _as_dict(payload.input)))
    return JobQueuedResponse(job_id=job_id)


@app.get("/v1/predict/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, user: dict[str, Any] = Depends(current_user)) -> JobStatusResponse:  # noqa: ARG001
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    result_obj = PredictResponse(**job["result"]) if job.get("result") else None
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=result_obj,
        error=job.get("error"),
    )


@app.get("/v1/me/history", response_model=HistoryResponse)
async def my_history(limit: int = 20, user: dict[str, Any] = Depends(current_user)) -> HistoryResponse:
    items = get_prediction_history(user_id=user["user_id"], limit=limit)
    return HistoryResponse(items=items)
