# tests/test_robustness.py

import pytest
import pandas as pd
import numpy as np
from fastapi import status

def test_prediction_with_missing_field(api_client, valid_serving_payload):
    """
    Test 1: API should return a 422 error if a required field is missing.
    """
    columns = valid_serving_payload['dataframe_split']['columns']
    data = valid_serving_payload['dataframe_split']['data']
    df = pd.DataFrame(data, columns=columns)

    if "EXT_SOURCE_2" in df.columns:
        df = df.drop(columns=["EXT_SOURCE_2"])
    else:
        pytest.skip("Skipping test: 'EXT_SOURCE_2' not found in example payload.")

    # FIX: Convert to the 'records' format expected by the API.
    malformed_payload = df.to_dict('records')

    response = api_client.post("/predict", json=malformed_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_prediction_with_incorrect_data_type(api_client, valid_serving_payload):
    """
    Test 2: API should return a 422 error if data has the wrong type.
    """
    columns = valid_serving_payload['dataframe_split']['columns']
    data = valid_serving_payload['dataframe_split']['data']
    df = pd.DataFrame(data, columns=columns)

    # Change a numerical feature to a string
    df.loc[0, "AMT_INCOME_TOTAL"] = "this-is-not-a-number"

    # FIX: Replace numpy.NaN with None for JSON compatibility before converting.
    # The .to_dict('records') method will then produce a clean list of dicts.
    df_safe = df.replace({np.nan: None})
    malformed_payload = df_safe.to_dict('records')
    
    response = api_client.post("/predict", json=malformed_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_prediction_with_empty_payload(api_client):
    """
    Test 3: API should return a 422 error for an empty request body.
    """
    # An empty JSON object {} is different from an empty list []
    response = api_client.post("/predict", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # A list of objects is the expected format, so an empty list should also be handled.
    response = api_client.post("/predict", json=[])
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_prediction_with_extra_field(api_client, valid_serving_payload):
    """
    Test 4: Pydantic should reject extra fields by default.
    """
    columns = valid_serving_payload['dataframe_split']['columns']
    data = valid_serving_payload['dataframe_split']['data']
    df = pd.DataFrame(data, columns=columns)

    df["EXTRA_FIELD_UNSEEN_BY_MODEL"] = "test_value"

    # FIX: Convert to the 'records' format expected by the API.
    payload_with_extra = df.to_dict('records')

    response = api_client.post("/predict", json=payload_with_extra)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY