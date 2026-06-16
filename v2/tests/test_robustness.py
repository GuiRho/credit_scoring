"""Robustness and error-handling tests for the serving API."""

import pytest
import pandas as pd
import numpy as np
from fastapi import status


def test_missing_field_rejected(api_client, valid_serving_payload) -> None:
    """Dropping a required feature should result in a 422 or 400 error."""
    cols = valid_serving_payload["dataframe_split"]["columns"]
    data = valid_serving_payload["dataframe_split"]["data"]
    df = pd.DataFrame(data, columns=cols)
    dropped = df.drop(columns=["EXT_SOURCE_2"], errors="ignore")
    if "EXT_SOURCE_2" in df.columns:
        payload = {"client_id": "1", "features": dropped.iloc[0].to_dict()}
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code in (
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
        )


def test_wrong_data_type_rejected(api_client, valid_serving_payload) -> None:
    """A string in a numeric field should be rejected."""
    cols = valid_serving_payload["dataframe_split"]["columns"]
    data = valid_serving_payload["dataframe_split"]["data"]
    df = pd.DataFrame(data, columns=cols).head(1)
    bad = df.replace({np.nan: None}).iloc[0].to_dict()
    bad["AMT_INCOME_TOTAL"] = "nan"
    resp = api_client.post("/predict", json={"client_id": "1", "features": bad})
    assert resp.status_code in (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_400_BAD_REQUEST,
    )


def test_empty_request_body(api_client) -> None:
    """Empty dict should return 422."""
    resp = api_client.post("/predict", json={})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_extra_field_rejected(api_client, valid_serving_payload) -> None:
    """Extra fields in features should be rejected by Pydantic or the model."""
    cols = valid_serving_payload["dataframe_split"]["columns"]
    data = valid_serving_payload["dataframe_split"]["data"]
    df = pd.DataFrame(data, columns=cols).head(1)
    with_extra = df.iloc[0].to_dict()
    with_extra["EXTRA_FIELD"] = "unseen"
    resp = api_client.post("/predict", json={"client_id": "1", "features": with_extra})
    assert resp.status_code in (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_400_BAD_REQUEST,
    )


def test_model_loading_failure() -> None:
    """Loading a non-existent model path should raise FileNotFoundError."""
    from credit_scoring.models import load_model
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path")
