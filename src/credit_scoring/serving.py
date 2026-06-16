"""FastAPI serving for credit scoring. Supports local model and MLflow registry modes."""

import os
from typing import Any
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "production_model")
MODEL_NAME = os.environ.get("MODEL_NAME", "credit_scoring_model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
USE_MLFLOW_REGISTRY = os.environ.get("USE_MLFLOW_REGISTRY", "").lower() in ("1", "true", "yes")

app = FastAPI(title="Credit Scoring API", version="2.0.0")
model: Any = None
mlflow_model_meta: dict | None = None


def load_model_local() -> Any:
    """Load model from local MLflow-packaged directory."""
    import yaml
    from pathlib import Path

    model_dir = Path(MODEL_DIR)
    mlmodel_path = model_dir / "MLmodel"
    if not mlmodel_path.exists():
        raise FileNotFoundError(f"MLmodel not found in {model_dir}")

    with open(mlmodel_path) as f:
        mlmodel = yaml.safe_load(f)

    global mlflow_model_meta
    mlflow_model_meta = mlmodel

    flavor = mlmodel.get("flavors", {}).get("sklearn")
    if not flavor:
        raise ValueError("No sklearn flavor found in MLmodel")

    return mlflow.sklearn.load_model(str(model_dir))


def load_model_registry() -> Any:
    """Load model from MLflow Model Registry."""
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not latest:
        raise ValueError(f"No model found for {MODEL_NAME}:{MODEL_STAGE}")

    run_id = latest[0].run_id
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@app.on_event("startup")
def startup() -> None:
    """Load model on startup."""
    global model
    try:
        model = load_model_registry() if USE_MLFLOW_REGISTRY else load_model_local()
    except Exception as e:
        model = None
        print(f"WARNING: Model failed to load at startup: {e}")


class ClientData(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    features: dict[str, Any] = Field(..., description="Feature name to value mapping")


class PredictionResponse(BaseModel):
    client_id: str
    probability: float
    prediction: int


@app.get("/")
def read_root() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "API is running", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData) -> dict[str, Any]:
    """Predict default probability for a single client."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.features])
        proba = model.predict_proba(df)[0]
        prob = float(proba[1]) if proba.ndim == 2 and proba.shape[1] >= 2 else float(proba[0])
        pred = int(prob >= 0.5)
        return {"client_id": data.client_id, "probability": prob, "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
