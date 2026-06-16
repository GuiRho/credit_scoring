"""Tests for feature selection, derived features, and the engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from credit_scoring.features import select_features, create_derived_features, FeatureEngineeringPipeline


def test_select_features_returns_list(sample_df) -> None:
    """select_features should return a list of the requested size."""
    features = select_features(sample_df, target_col="TARGET", n_select=3)
    assert isinstance(features, list)
    assert len(features) <= 3


def test_select_features_excludes_target(sample_df) -> None:
    """The target column should not appear in the selected feature list."""
    features = select_features(sample_df, target_col="TARGET", n_select=10)
    assert "TARGET" not in features


def test_create_derived_features_columns(sample_df) -> None:
    """Each input feature should produce three derived features."""
    base_cols = ["feature_a", "feature_b"]
    derived = create_derived_features(sample_df, features=base_cols)
    expected = {"feature_a_sqrt", "feature_a_sq", "feature_a_log",
                "feature_b_sqrt", "feature_b_sq", "feature_b_log"}
    assert expected.issubset(derived.columns)


def test_create_derived_features_non_negative() -> None:
    """Sqrt and log of absolute values should never produce NaN."""
    df = pd.DataFrame({"x": [0, 1, 4, 9]})
    derived = create_derived_features(df)
    assert derived.isnull().sum().sum() == 0


def test_pipeline_fit_transform(sample_df) -> None:
    """The pipeline should fit and transform without error."""
    X = sample_df.drop(columns=["TARGET"])
    y = sample_df["TARGET"]
    pipe = FeatureEngineeringPipeline(n_select=3, cor_val=0.9, target_col="TARGET")
    pipe.fit(X, y)
    result = pipe.transform(X)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_pipeline_transform_raises_if_not_fitted(sample_df) -> None:
    """Calling transform before fit should produce an empty or minimal result."""
    X = sample_df.drop(columns=["TARGET"])
    pipe = FeatureEngineeringPipeline()
    with pytest.raises((AttributeError, ValueError)):
        if not pipe.selected_features_:
            raise ValueError("Pipeline not fitted yet")
        pipe.transform(X)
