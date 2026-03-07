"""AI inference service with async wrappers for API responsiveness."""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Any

from ..config import MODEL_PATH
from ..predict import load_model, predict_single


@lru_cache(maxsize=1)
def get_model():
    return load_model(MODEL_PATH)


def _predict_sync(input_payload: dict[str, Any]) -> dict[str, Any]:
    model = get_model()
    label, probability = predict_single(model, input_payload)
    return {"label": label, "probability": float(probability)}


async def predict_async(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Run CPU-bound prediction in a worker thread to keep event loop responsive."""
    start = time.perf_counter()
    result = await asyncio.to_thread(_predict_sync, input_payload)
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 2)
    return result
