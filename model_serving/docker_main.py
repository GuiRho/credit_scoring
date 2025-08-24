""""
separate main file for the Docker deployment that loads the model from the packaged folder

Create a copy of model_serving/main.py and name it model_serving/docker_main.py.
Modify docker_main.py to load the model from the local /model directory (which is where we'll place it in the Docker container).

Note: This docker_main.py still needs to connect to your MLflow server to get the run parameters. We will address this in the Dockerfile.
"""""

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
# --- NEW MODEL LOADING LOGIC FOR DOCKER ---
def load_packaged_model():
    """Loads the model, threshold, and features from a local directory."""
    global model, BEST_THRESHOLD, EXPECTED_FEATURES
    
    # In Docker, the model will be at a fixed path '/model'
    model_path = "/model" 
    
    try:
        # Load the model from the local directory
        model = mlflow.pyfunc.load_model(model_path)
        
        # Get metadata from the MLmodel file in that directory
        model_info = mlflow.models.get_model_info(model_path)

        # Get the threshold from the run parameters linked in the model
        client = mlflow.tracking.MlflowClient()
        run_id = model_info.run_id
        run_data = client.get_run(run_id).data
        
        threshold_str = run_data.params.get("best_threshold")
        if threshold_str:
            BEST_THRESHOLD = float(threshold_str)
        else:
            raise RuntimeError("'best_threshold' parameter not found in the model's associated run.")
            
        if model_info.signature:
            EXPECTED_FEATURES = [spec.name for spec in model_info.signature.inputs]
        else:
            raise RuntimeError("Model signature not found.")

        print("--- Packaged model loaded successfully ---")
        print(f"Model path: {model_path}")
        print(f"Run ID: {run_id}")
        print(f"Best Threshold: {BEST_THRESHOLD}")
        print("------------------------------------------")

    except Exception as e:
        print(f"FATAL: Error loading packaged model: {e}")
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

#  the startup event call the new function
@app.on_event("startup")
async def startup_event():
    load_packaged_model()


# The rest of your main.py remains the same!
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(client_id: str, data: ClientData):
    print(f"Received features keys: {list(data.features.keys())}")  # Debug print
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