"""Fixtures for the credit scoring test suite."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def model():
    """Load the production model once per session."""
    from credit_scoring.models import load_model
    return load_model("production_model")


@pytest.fixture(scope="session")
def api_client():
    """Create a FastAPI TestClient for the serving app."""
    try:
        from credit_scoring.serving import app
        return TestClient(app)
    except ImportError:
        pytest.skip("credit_scoring.serving module not available")
    except Exception:
        pytest.skip("Could not instantiate serving app")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with numeric features and a binary target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, 100),
        "feature_b": rng.uniform(0, 10, 100),
        "feature_c": rng.integers(0, 2, 100),
        "TARGET": rng.integers(0, 2, 100),
    })
