import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback
import numpy as np

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "CreditScoringRF"
MODEL_STAGE = "Production"
HOST = "127.0.0.1"
PORT = 8000

# --- MLflow Setup ---
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- FastAPI App ---
app = FastAPI(
    title="Credit Scoring API",
    description="API for predicting credit default risk using a trained ML model.",
    version="1.0.0",
)

# --- Model Loading ---
model = None
model_signature = None
model_features = None


@app.on_event("startup")
def load_model():
    """
    Load the model from the MLflow Model Registry on application startup.
    """
    global model, model_signature, model_features
    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")

        model_info = mlflow.models.get_model_info(model_uri)
        if model_info.signature:
            model_signature = model_info.signature
            model_features = [input.name for input in model_info.signature.inputs]
            print(f"Model features loaded ({len(model_features)}): {model_features}")
        else:
            raise RuntimeError("Model signature with features not found.")

    except Exception as e:
        print(f"CRITICAL: Error loading model during startup: {e}")
        model = None


# --- Pydantic Model for Input ---
class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]


# --- API Endpoints ---
@app.get("/health", summary="Check API Health")
def health_check():
    """
    Endpoint to check if the API is running and if the model is loaded.
    """
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        return {
            "status": "error",
            "model_loaded": False,
            "message": "Model could not be loaded at startup.",
        }


@app.post("/predict", summary="Get Credit Score Prediction")
def predict(payload: PredictionInput):
    """
    Endpoint to get a credit score prediction.
    """
    if not model:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    print("--- New Prediction Request ---")
    try:
        print(f"Received payload with {len(payload.data)} records.")

        input_df = pd.DataFrame(payload.data)
        print("Successfully converted payload to DataFrame.")

        # --- Data Cleaning and Type Conversion ---
        for feature in model_signature.inputs:
            feature_name = feature.name
            feature_type = feature.type.to_numpy()

            if feature_name == "CODE_GENDER":
                # Explicitly map CODE_GENDER to numerical values
                input_df[feature_name] = input_df[feature_name].map({'M': 0, 'F': 1, 'XNA': 2}).fillna(0)
            elif np.issubdtype(feature_type, np.bool_):
                # For boolean features, convert to boolean, coerce errors, and fill NaNs with False
                input_df[feature_name] = input_df[feature_name].astype(bool)
            elif np.issubdtype(feature_type, np.integer):
                # For integer features, convert to numeric, coerce errors, fill NaNs, and convert to int
                input_df[feature_name] = pd.to_numeric(input_df[feature_name], errors='coerce').fillna(0).astype(int)
            elif np.issubdtype(feature_type, np.number):
                # For other numerical features (float), convert to numeric, coerce errors, and fill NaNs
                input_df[feature_name] = pd.to_numeric(input_df[feature_name], errors='coerce').fillna(0)
            else:
                # For other types (e.g., string/categorical), fill missing with a placeholder
                input_df[feature_name] = input_df[feature_name].fillna('')

        print("DataFrame after cleaning and type conversion:")
        print(input_df.head(1).to_string())
        print(input_df.info())

        # Detailed feature validation
        incoming_features = set(input_df.columns)
        expected_features = set(model_features)

        if incoming_features != expected_features:
            missing = sorted(list(expected_features - incoming_features))
            extra = sorted(list(incoming_features - expected_features))
            error_detail = {}
            if missing:
                error_detail["missing_features"] = missing
                print(f"ERROR: Missing features: {missing}")
            if extra:
                error_detail["extra_features"] = extra
                print(f"ERROR: Extra features: {extra}")
            raise HTTPException(status_code=400, detail=error_detail)

        print("All expected features are present. Reordering columns.")
        input_df = input_df[model_features]

        # --- Debugging before prediction ---
        print("\n--- Debugging input_df before model.predict() ---")
        print("input_df.dtypes:")
        print(input_df.dtypes)
        print("\ninput_df.head():")
        print(input_df.head())
        print("\nmodel_features (expected order and names):")
        print(model_features)
        print("\nmodel_signature (full signature details):")
        print(model_signature)
        print("-------------------------------------------")

        print("Calling model.predict()...")
        predictions = model.predict(input_df)
        print(f"Prediction successful. Result: {predictions}")

        return {"predictions": predictions.tolist()}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)