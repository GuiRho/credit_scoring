"""Tests for model loading, prediction shape, consistency, and edge cases."""

import pytest
import pandas as pd
import numpy as np


def test_model_loading(model) -> None:
    """Verify the loaded model has predict and predict_proba methods."""
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_prediction_output_shape(model, valid_input_data) -> None:
    """Ensure predict and predict_proba return expected shapes."""
    df = pd.DataFrame(valid_input_data["data"], columns=valid_input_data["columns"])
    pred = model.predict(df)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (5,), f"Expected (5,), got {pred.shape}"
    proba = model.predict_proba(df)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (5, 2), f"Expected (5, 2), got {proba.shape}"
    assert np.all(np.isclose(np.sum(proba, axis=1), 1.0))


def test_prediction_consistency(model, valid_input_data) -> None:
    """Verify the model returns a known expected probability."""
    df = pd.DataFrame(valid_input_data["data"], columns=valid_input_data["columns"])
    proba = model.predict_proba(df)
    assert proba[0, 1] == pytest.approx(0.03918383094454616, abs=0.001)


def test_prediction_with_empty_df(model) -> None:
    """Edge case: predict with an empty DataFrame (0 rows)."""
    empty_df = pd.DataFrame()
    with pytest.raises(Exception):
        model.predict(empty_df)


def test_prediction_with_wrong_column_types(model) -> None:
    """Edge case: predict with non-numeric column values."""
    bad_df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
    with pytest.raises(Exception):
        model.predict(bad_df)
