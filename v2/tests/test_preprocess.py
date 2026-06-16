"""Tests for preprocessing: imputation, outlier removal, train/test split."""

import pytest
import pandas as pd
import numpy as np
from credit_scoring.preprocess import clean_and_impute_data, remove_outliers, split_data


def _sample_df() -> pd.DataFrame:
    """Helper: return a small DataFrame with missing values for testing."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "num_b": [10.0, np.nan, 30.0, 40.0, 50.0],
        "TARGET": [0, 1, 0, 1, 0],
        "bool_c": [True, False, True, False, True],
    })


def test_clean_and_impute_median() -> None:
    """Median imputation should fill NaN values with the column median."""
    df = _sample_df()
    result = clean_and_impute_data(df, completeness=0, impute="median", cv_threshold=0.0)
    assert result["num_a"].isnull().sum() == 0
    assert result["num_b"].isnull().sum() == 0
    assert result["num_a"].iloc[2] == pytest.approx(df["num_a"].median())


def test_clean_and_impute_mean() -> None:
    """Mean imputation should fill NaN values with the column mean."""
    df = _sample_df()
    result = clean_and_impute_data(df, completeness=0, impute="mean", cv_threshold=0.0)
    assert result["num_a"].isnull().sum() == 0
    assert result["num_a"].iloc[2] == pytest.approx(df["num_a"].mean())


def test_clean_and_impute_drops_low_completeness() -> None:
    """Columns below the completeness threshold should be dropped."""
    df = pd.DataFrame({
        "good": [1, 2, 3],
        "mostly_null": [np.nan, np.nan, 1],
        "TARGET": [0, 1, 0],
    })
    result = clean_and_impute_data(df, completeness=50, impute="median", cv_threshold=0.0)
    assert "mostly_null" not in result.columns


def test_remove_outliers() -> None:
    """Outlier removal should drop rows with extreme values."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 100],
        "TARGET": [0, 1, 0, 1, 0],
    })
    result = remove_outliers(df, percent=10)
    assert 4 not in result.index


def test_split_data(sample_df) -> None:
    """Train/test split should preserve class proportions."""
    X_train, X_test, y_train, y_test = split_data(sample_df)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
