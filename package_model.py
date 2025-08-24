import os
import shutil
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import logging

# --- Configuration ---
MODEL_NAME = "credit_scoring"
MODEL_STAGE = "Production"
OUTPUT_PATH = "production_model"
tracking_uri = "file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_production_model(output_path: str):
    """
    Finds the production model, downloads its files, and saves them
    to a specified local path.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Downloading production model from: {model_uri}")

        # Check if the output directory exists and clear it
        if os.path.exists(output_path):
            logger.info(f"Clearing existing directory: {output_path}")
            shutil.rmtree(output_path)

        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Download the model artifacts
        local_path = ModelsArtifactRepository(model_uri).download_artifacts(
            artifact_path="",  # Download all artifacts from the model's root
            dst_path=output_path
        )

        logger.info(f"Production model successfully downloaded to: {local_path}")
        logger.info("Directory contents: %s", os.listdir(local_path))

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

if __name__ == "__main__":
    download_production_model(OUTPUT_PATH)
