"""Tests for the FastAPI serving endpoint (was test_dashboard_api.py)."""

import pytest
import pandas as pd
from fastapi import status


def test_health_check(api_client) -> None:
    """Health-check endpoint should return 200 and indicate running status."""
    response = api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "API is running"


def test_successful_prediction(api_client, valid_serving_payload) -> None:
    """A valid payload should return a prediction with probability and label."""
    cols = valid_serving_payload["dataframe_split"]["columns"]
    data = valid_serving_payload["dataframe_split"]["data"]
    df = pd.DataFrame(data, columns=cols).head(1)
    payload = {
        "client_id": "999999",
        "features": df.iloc[0].to_dict(),
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert isinstance(body["prediction"], int)
    assert isinstance(body["probability"], float)
    assert 0.0 <= body["probability"] <= 1.0


def test_invalid_input(api_client) -> None:
    """A request with a string instead of a number should be rejected."""
    payload = {
        "client_id": "1",
        "features": {"AMT_INCOME_TOTAL": "not-a-number"},
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code in (status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST)


def test_missing_client_id(api_client) -> None:
    """Omitting the required client_id field should return 422."""
    response = api_client.post("/predict", json={"features": {"a": 1}})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_empty_payload(api_client) -> None:
    """An empty request body should return 422."""
    response = api_client.post("/predict", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
