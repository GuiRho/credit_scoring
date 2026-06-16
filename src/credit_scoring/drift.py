"""Data drift analysis for credit scoring using Evidently.

Compares a reference (training) dataset against a current (test) dataset
and produces an HTML drift report via ``DataDriftPreset``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for data drift analysis."""

    dataset_dir: str | Path
    output_dir: str | Path
    report_filename: str = "data_drift_report.html"


def load_datasets(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference (train) and current (test) parquet datasets.

    Returns:
        Tuple of (reference_df, current_df).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train_processed.parquet"
    test_path = data_dir / "test_processed.parquet"

    if not train_path.is_file():
        raise FileNotFoundError(f"train_processed.parquet not found in {data_dir}")
    reference = pd.read_parquet(train_path)
    logger.info("Reference data: %s", reference.shape)

    if not test_path.is_file():
        raise FileNotFoundError(f"test_processed.parquet not found in {data_dir}")
    current = pd.read_parquet(test_path)
    logger.info("Current data: %s", current.shape)

    return reference, current


def generate_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str | Path
) -> Path:
    """Generate and save an Evidently data-drift HTML report.

    Args:
        reference: Baseline dataset (training).
        current: Dataset under test.
        output_path: Destination path for the HTML report.

    Returns:
        Resolved ``Path`` to the saved report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))

    logger.info("Drift report saved: %s", output_path.resolve())
    return output_path


def run_drift_analysis(config: DriftConfig) -> Path:
    """Run the full drift analysis pipeline.

    Args:
        config: Drift configuration.

    Returns:
        Path to the generated HTML report.
    """
    reference, current = load_datasets(config.dataset_dir)
    report_path = Path(config.output_dir) / config.report_filename
    return generate_report(reference, current, report_path)
