import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# --- CONFIGURATION ---
# Define the name of the model in the MLflow Model Registry
MODEL_NAME = "credit_scoring" # You used "credit_scoring"
MODEL_STAGE = "Production"

# Set the MLflow tracking URI
tracking_uri = "file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"
mlflow.set_tracking_uri(tracking_uri)

# --- GLOBAL VARIABLES ---
model = None
BEST_THRESHOLD = 0.5
EXPECTED_FEATURES = []

# --- IN model_serving/main.py ---

def load_production_model():
    """Loads the production model, its threshold, and feature list."""
    global model, BEST_THRESHOLD, EXPECTED_FEATURES
    
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        
        # Load the model for prediction
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get the model's high-level info
        model_info = mlflow.models.get_model_info(model_uri)
        
        # --- NEW AND IMPROVED LOGIC ---
        # 1. Get the Run ID that generated this model version
        run_id = model_info.run_id
        if not run_id:
            raise RuntimeError("Model version is not linked to a Run ID.")

        # 2. Use the MlflowClient to get the full details of that run
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data

        # 3. Reliably get the threshold from the run's parameters
        threshold_str = run_data.params.get("best_threshold")
        if threshold_str:
            BEST_THRESHOLD = float(threshold_str)
        else:
            print("CRITICAL WARNING: 'best_threshold' not found in run parameters. Falling back to 0.5.")
            BEST_THRESHOLD = 0.5
        # --- END OF NEW LOGIC ---

        # Get the expected features from the signature
        if model_info.signature:
            EXPECTED_FEATURES = [spec.name for spec in model_info.signature.inputs]
        else:
            raise RuntimeError("Model signature not found in MLflow model info.")

        print("--- Production model loaded successfully ---")
        print(f"Model: {MODEL_NAME}, Stage: {MODEL_STAGE}")
        print(f"Run ID: {run_id}")
        print(f"Best Threshold: {BEST_THRESHOLD}") # Should now be correct
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
    version="1.1.0" # Version bump!
)

# Pydantic models (no changes here)
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

# The rest of your main.py remains the same!
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(client_id: str, data: ClientData):
    if not model or not EXPECTED_FEATURES:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        input_df = pd.DataFrame([data.features])
        missing_features = set(EXPECTED_FEATURES) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {list(missing_features)}")
        
        input_df = input_df[EXPECTED_FEATURES]
        
        probability = model.predict(input_df)[0][1]
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