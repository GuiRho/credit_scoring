"""Unified configuration for the credit scoring package."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")
MODEL_DIR = Path("production_model")
OUTPUT_DIR = Path("output")


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load and return a JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)


def get_project_root() -> Path:
    """Find and return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent
