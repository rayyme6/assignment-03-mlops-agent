"""FastAPI inference: liveness is distinct from model readiness."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from training.train import FEATURES


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    if path.is_file():
        # Only load artifacts produced by this trusted training pipeline.
        app.state.model = joblib.load(path)
    yield


app = FastAPI(title="Iris inference", lifespan=lifespan)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[float] = Field(min_length=4, max_length=4, allow_inf_nan=False)


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(503, "Model is not loaded")
    return {"status": "ready", "version": os.getenv("MODEL_VERSION", "unknown"),
            "run_id": os.getenv("MODEL_RUN_ID", "unknown")}


@app.post("/predict")
def predict(payload: Prediction):
    ready()
    if any(x <= 0 or x > 30 for x in payload.features):
        raise HTTPException(422, "Measurements must be in (0,30] centimetres")
    frame = pd.DataFrame([payload.features], columns=FEATURES)
    return {"class": int(app.state.model.predict(frame)[0]),
            "run_id": os.getenv("MODEL_RUN_ID", "unknown")}
