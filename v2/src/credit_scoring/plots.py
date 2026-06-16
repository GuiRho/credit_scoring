"""SHAP visualization and export utilities.

Renders summary plots, dependency plots, and saves
global feature importance as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def generate_summary_plots(
    shap_values: list[np.ndarray],
    X_processed: pd.DataFrame,
    class_names: list[str],
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    """Generate SHAP summary plots — one per class."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(class_names):
        fig, _ = plt.subplots()
        shap.summary_plot(shap_values[i], X_processed, max_display=20, show=False)
        plt.title(f"SHAP Summary Plot — {name}", fontsize=16)
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        path = output_dir / f"summary_plot_{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        logger.info("Saved summary plot: %s", path)


def generate_dependency_plots(
    shap_values: list[np.ndarray],
    X_processed: pd.DataFrame,
    features: list[str],
    class_names: list[str],
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    """Generate SHAP dependency plots — one per feature per class."""
    base_dir = Path(output_dir) / "dependency_plots"

    for class_idx, class_name in enumerate(class_names):
        class_dir = base_dir / class_name.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        for feature in features:
            fig, _ = plt.subplots()
            shap.dependence_plot(feature, shap_values[class_idx], X_processed, show=False)
            plt.title(f"SHAP Dependence: {feature} ({class_name})", fontsize=14)
            plt.tight_layout()
            safe = "".join(c for c in feature if c.isalnum() or c in ("_", "-")).rstrip()
            path = class_dir / f"dependency_{safe}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            logger.info("Saved dependency plot: %s", path)


def save_global_shap_importance(
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Export mean absolute SHAP importance for the positive class to JSON."""
    pos_idx = 1 if len(class_names) >= 2 else 0
    mean_abs = np.abs(shap_values[pos_idx]).mean(axis=0)

    records = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .to_dict(orient="records")
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Global feature importance saved: %s", out)
