import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np

# --- CONFIGURATION ---
# Define the name of the model in the MLflow Model Registry
MODEL_NAME = "credit_scoring"
MODEL_STAGE = "Production"

# Set the MLflow tracking URI
tracking_uri = "file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"
mlflow.set_tracking_uri(tracking_uri)

# --- GLOBAL VARIABLES ---
model = None
BEST_THRESHOLD = 0.5
EXPECTED_FEATURES = []

def load_production_model():
    """Loads the production model, its threshold, and feature list."""
    global model, BEST_THRESHOLD, EXPECTED_FEATURES
    
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        
        # Load the model for prediction
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get the model's high-level info
        model_info = mlflow.models.get_model_info(model_uri)
        
        # Get the Run ID that generated this model version
        run_id = model_info.run_id
        if not run_id:
            raise RuntimeError("Model version is not linked to a Run ID.")

        # Use the MlflowClient to get the full details of that run
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data

        # Reliably get the threshold from the run's parameters
        threshold_str = run_data.params.get("best_threshold")
        if threshold_str:
            BEST_THRESHOLD = float(threshold_str)
        else:
            print("CRITICAL WARNING: 'best_threshold' not found in run parameters. Falling back to 0.5.")
            BEST_THRESHOLD = 0.5

        # Get the expected features from the signature
        if model_info.signature:
            EXPECTED_FEATURES = [spec.name for spec in model_info.signature.inputs]
        else:
            raise RuntimeError("Model signature not found in MLflow model info.")

        print("--- Production model loaded successfully ---")
        print(f"Model: {MODEL_NAME}, Stage: {MODEL_STAGE}")
        print(f"Run ID: {run_id}")
        print(f"Best Threshold: {BEST_THRESHOLD}")
        print(f"Model expects {len(EXPECTED_FEATURES)} features.")
        print("------------------------------------------")

    except Exception as e:
        print(f"FATAL: Error loading production model: {e}")
        print("Ensure a model with the correct name has been transitioned to the 'Production' stage in MLflow.")
        exit()

# --- API DEFINITION ---
app = FastAPI(
    title="Credit Scoring API",
    description="API to predict client credit default probability from the production model.",
    version="1.1.0"
)

# Pydantic models
class ClientData(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    client_id: str
    probability: float
    prediction: int
    threshold: float

# --- API Events and Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    load_production_model()

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(client_id: str, data: ClientData):
    print(f"Received features keys: {list(data.features.keys())}")
    if not model or not EXPECTED_FEATURES:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        input_df = pd.DataFrame([data.features])
        missing_features = set(EXPECTED_FEATURES) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {list(missing_features)}")
        
        input_df = input_df[EXPECTED_FEATURES]
        
        # Get prediction from model
        prediction_output = model.predict(input_df)
        
        # Handle different output formats
        if isinstance(prediction_output, (pd.DataFrame, pd.Series)):
            prediction_output = prediction_output.values
            
        if isinstance(prediction_output, np.ndarray):
            # If it's a 2D array with probabilities for both classes
            if prediction_output.ndim == 2 and prediction_output.shape[1] >= 2:
                probability = prediction_output[0, 1]  # Get probability for positive class
            # If it's a 1D array with probabilities
            elif prediction_output.ndim == 1:
                probability = prediction_output[0]
            else:
                raise ValueError(f"Unexpected prediction output shape: {prediction_output.shape}")
        else:
            # If it's a scalar
            probability = float(prediction_output)
        
        prediction = 1 if probability >= BEST_THRESHOLD else 0
        
        return {
            "client_id": client_id,
            "probability": probability,
            "prediction": prediction,
            "threshold": BEST_THRESHOLD
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")