"""SHAP-based model analysis for credit scoring.

Provides the core SHAP computation pipeline:
- Load a packaged MLflow model + test data
- Stratified sampling
- Extract pipeline internals (preprocessor + classifier)
- Compute SHAP values via TreeExplainer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from credit_scoring.plots import generate_dependency_plots, generate_summary_plots, save_global_shap_importance

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for SHAP analysis."""

    model_dir: str | Path
    dataset_dir: str | Path
    output_dir: str | Path
    sample_size: int = 500
    top_n_features: int = 8
    plot_dpi: int = 150
    random_state: int = 42


def load_packaged_model(model_path: str | Path) -> tuple[Any, Any, float]:
    """Load an MLflow pyfunc model and parse its MLmodel metadata.

    Returns:
        Tuple of (model, mlmodel_data, best_threshold).
    """
    model_path = Path(model_path)
    mlmodel_path = model_path / "MLmodel"

    if not mlmodel_path.is_file():
        raise FileNotFoundError(f"MLmodel not found at {mlmodel_path.resolve()}")

    model = mlflow.pyfunc.load_model(str(model_path))
    with open(mlmodel_path) as f:
        mlmodel_data = yaml.safe_load(f)

    metadata = mlmodel_data.get("metadata", {})
    best_threshold = float(metadata.get("best_threshold", 0.5))

    logger.info("Model loaded from %s | best_threshold=%.3f", model_path, best_threshold)
    return model, mlmodel_data, best_threshold


def load_test_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed test parquet and split into X / y."""
    data_dir = Path(data_dir)
    test_path = data_dir / "test_processed.parquet"

    if not test_path.is_file():
        raise FileNotFoundError(f"test_processed.parquet not found in {data_dir}")

    df = pd.read_parquet(test_path)
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    logger.info("Test data loaded: X=%s, y=%s", X.shape, y.shape)
    return X, y


def sample_data(
    X: pd.DataFrame, y: pd.Series, sample_size: int, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified sample from the test set."""
    actual_size = min(sample_size, len(X))
    if actual_size < sample_size:
        logger.warning("Requested %d samples, only %d available", sample_size, actual_size)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=actual_size, random_state=random_state)
    _, idx = next(splitter.split(X, y))
    X_sampled = X.iloc[idx]
    y_sampled = y.iloc[idx]
    logger.info("Sampled %d rows (stratified)", actual_size)
    return X_sampled, y_sampled


def extract_pipeline(model: Any, X_sample: pd.DataFrame) -> tuple[Any, Any, np.ndarray, pd.DataFrame]:
    """Extract preprocessor, classifier, feature names, and transform sample.

    Assumes the model's sklearn pipeline has two steps: (preprocessor, classifier).
    """
    pipeline = model._model_impl.sklearn_model  # noqa: SLF001
    preprocessor = pipeline.steps[0][1]
    classifier = pipeline.steps[1][1]

    feature_names = preprocessor.get_feature_names_out(input_features=X_sample.columns)
    X_processed = pd.DataFrame(
        preprocessor.transform(X_sample),
        index=X_sample.index,
        columns=feature_names,
    )
    logger.info("Pipeline extracted — %d features, classifier=%s", len(feature_names), type(classifier).__name__)
    return preprocessor, classifier, feature_names, X_processed


def compute_shap_values(
    classifier: Any, X_processed: pd.DataFrame
) -> list[np.ndarray]:
    """Compute SHAP values using TreeExplainer.

    Returns:
        List of 2-D arrays, one per class.
    """
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    if not isinstance(shap_values, list) or len(shap_values) != len(classifier.classes_):
        raise TypeError(
            f"Expected {len(classifier.classes_)} SHAP arrays, got "
            f"{type(shap_values)} of len {len(shap_values) if isinstance(shap_values, list) else 'N/A'}"
        )

    logger.info("SHAP values computed — %d classes", len(shap_values))
    return shap_values


def get_top_features_by_shap(
    shap_values: list[np.ndarray], feature_names: list[str], n_features: int
) -> list[str]:
    """Rank top features by mean absolute SHAP value across all classes."""
    global_importance = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    mean_abs = np.mean(global_importance, axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
    )
    top = importance_df.head(n_features)["feature"].tolist()
    logger.info("Top %d features: %s", n_features, top)
    return top


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Orchestrate the full SHAP analysis pipeline.

    Returns:
        Dict with paths to generated artifacts.
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = load_packaged_model(config.model_dir)
    X_test, y_test = load_test_data(config.dataset_dir)
    X_sample, y_sample = sample_data(X_test, y_test, config.sample_size, config.random_state)

    sample_df = pd.concat([X_sample, y_sample], axis=1)
    csv_path = out_dir / "data_for_analysis.csv"
    sample_df.to_csv(csv_path, index=False)
    logger.info("Sample saved: %s", csv_path)

    _, classifier, feature_names, X_processed = extract_pipeline(model, X_sample)
    shap_values = compute_shap_values(classifier, X_processed)

    class_names = [f"Class_{c}" for c in classifier.classes_]
    top_features = get_top_features_by_shap(shap_values, feature_names, config.top_n_features)

    plots_dir = out_dir / "plots"
    generate_summary_plots(shap_values, X_processed, class_names, plots_dir, config.plot_dpi)
    generate_dependency_plots(shap_values, X_processed, top_features, class_names, plots_dir, config.plot_dpi)

    importance_path = Path(config.model_dir) / "global_feature_importance.json"
    save_global_shap_importance(shap_values, feature_names, class_names, importance_path)

    return {
        "sample_csv": str(csv_path.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "importance_json": str(importance_path.resolve()),
    }
