"""FastAPI inference service; a failed model load makes readiness fail, not liveness."""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

LOG = logging.getLogger(__name__)
LABELS = ["setosa", "versicolor", "virginica"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[list[float]] = Field(min_length=1, max_length=1000)

    @field_validator("instances")
    @classmethod
    def valid_features(cls, rows: list[list[float]]) -> list[list[float]]:
        if any(len(row) != 4 or not all(math.isfinite(x) and x >= 0 for x in row) for row in rows):
            raise ValueError("Each instance needs four finite, nonnegative Iris measurements")
        return rows


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.metadata = {}
        try:
            metadata = json.loads(path.with_name("metadata.json").read_text())
            if metadata.get("quality_gate") != "passed":
                raise ValueError("Model has not passed its quality gate")
            model = joblib.load(path)
            if getattr(model, "n_features_in_", None) != 4:
                raise ValueError("Model feature count is incompatible")
            app.state.model = model
            app.state.metadata = metadata
            LOG.info("Approved model loaded: run_id=%s", metadata.get("run_id"))
        except Exception:
            LOG.warning("Model unavailable; readiness disabled")
        yield

    app = FastAPI(title="Iris Prediction API", version="1.0.0", lifespan=lifespan)

    def ready() -> None:
        if getattr(app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="No approved model is loaded")

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness() -> dict:
        ready()
        return {"status": "ready", "run_id": app.state.metadata.get("run_id")}

    @app.get("/version")
    def version() -> dict:
        ready()
        metadata = app.state.metadata
        return {"application_version": os.getenv("APP_VERSION", "1.0.0"),
                **{key: metadata.get(key) for key in
                   ("run_id", "model_version", "git_commit", "dvc_hash", "trained_at", "quality_gate")}}

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict:
        ready()
        features = app.state.metadata.get("features")
        frame = pd.DataFrame(request.instances, columns=features)
        predictions = [int(value) for value in app.state.model.predict(frame)]
        return {"predictions": predictions, "labels": [LABELS[value] for value in predictions],
                "model_version": app.state.metadata.get("model_version")}

    return app


app = create_app()
