import pandas as pd
import mlflow
import yaml  
from pathlib import Path 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# --- CONFIGURATION (No changes) ---
MODEL_NAME = "credit_scoring"
MODEL_STAGE = "Production"

# --- GLOBAL VARIABLES (No changes) ---
model = None
BEST_THRESHOLD = 0.5
EXPECTED_FEATURES = []

# --- NEW, ROBUST MODEL LOADING LOGIC ---
def load_packaged_model():
    """
    Loads the model and its metadata by directly parsing the MLmodel file.
    This is fully self-contained and has no dependency on the MLflow tracking client.
    """
    global model, BEST_THRESHOLD, EXPECTED_FEATURES
    
    model_path = Path("model")
    mlmodel_path = model_path / "MLmodel"

    try:
        # 1. Load the model for making predictions
        model = mlflow.pyfunc.load_model(str(model_path))
        
        # 2. Manually parse the MLmodel YAML file for metadata
        with open(mlmodel_path, 'r') as f:
            mlmodel_data = yaml.safe_load(f)

        # 3. Get the threshold from the custom metadata
        custom_metadata = mlmodel_data.get("metadata", {})
        if 'best_threshold' in custom_metadata:
            BEST_THRESHOLD = float(custom_metadata['best_threshold'])
        else:
            raise RuntimeError("'best_threshold' not found in MLmodel custom metadata.")

        # 4. Get expected features from the signature
        signature = mlmodel_data.get("signature", {})
        if signature and 'inputs' in signature:
            # The input is a JSON string, so we need to parse it
            inputs_list = yaml.safe_load(signature['inputs'])
            EXPECTED_FEATURES = [spec['name'] for spec in inputs_list]
        else:
            raise RuntimeError("Model signature not found in MLmodel.")

        print("--- Packaged model loaded successfully (Direct YAML Parse) ---")
        print(f"Model path: {model_path}")
        print(f"Best Threshold: {BEST_THRESHOLD}")
        print(f"Model expects {len(EXPECTED_FEATURES)} features.")
        print("---------------------------------------------------------")

    except Exception as e:
        # This will print the exact error to Cloud Run logs for easier debugging
        print(f"FATAL: Error loading packaged model: {e}")
        # The exit() call ensures the container stops if the model can't be loaded
        exit(1)


# --- API DEFINITION (No changes) ---
app = FastAPI(
    title="Credit Scoring API",
    description="API to predict client credit default probability from the production model.",
    version="1.2.0" # Version bump!
)

class ClientData(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    client_id: str
    probability: float
    prediction: int
    threshold: float

@app.on_event("startup")
async def startup_event():
    load_packaged_model()

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
        # The model returns probabilities for class 0 and 1, we need the probability for class 1 (default)
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