"""Model loading and MLflow registry utilities."""

import pickle
import logging
from pathlib import Path
from typing import Any, Optional
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

logger = logging.getLogger(__name__)


def load_model(model_dir: str | Path) -> Any:
    """Load a pickled model from the given directory."""
    model_path = Path(model_dir) / "model.pkl"
    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def download_from_registry(
    model_name: str,
    version: Optional[str] = None,
    stage: str = "Production",
    dst_path: str | Path = "production_model",
    tracking_uri: Optional[str] = None,
) -> Path:
    """Download a model from the MLflow registry to a local directory.

    Args:
        model_name: Name of the registered model.
        version: Specific version string. Overrides *stage* if set.
        stage: Model stage alias (ignored if *version* is given).
        dst_path: Local destination directory.
        tracking_uri: MLflow tracking URI.

    Returns:
        Path to the local directory containing the downloaded artifacts.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    uri = f"models:/{model_name}/{version or stage}"
    logger.info("Downloading model from %s", uri)
    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)
    local = ModelsArtifactRepository(uri).download_artifacts(
        artifact_path="", dst_path=str(dst)
    )
    return Path(local)
