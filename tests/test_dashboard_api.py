# tests/test_dashboard_api.py

import pytest
import pandas as pd
from fastapi import status

def test_api_health_check(api_client):
    """
    Test 1: Check if the root endpoint is available (health check).
    """
    response = api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert "status" in response_json
    assert response_json["status"] == "API is running"

def test_successful_prediction(api_client, valid_serving_payload):
    """
    Test 2: Check a successful prediction via the /predict endpoint.
    """
    # Convert the payload to a DataFrame to easily manipulate it.
    columns = valid_serving_payload['dataframe_split']['columns']
    data = valid_serving_payload['dataframe_split']['data']
    df = pd.DataFrame(data, columns=columns)
    
    # An endpoint requiring a `client_id` typically processes one client at a time.
    # We will send only the first row of data for this test.
    records_payload = df.head(1).to_dict('records')

    # FIX: Add the required 'client_id' as a query parameter.
    # We can use a dummy value like 999999 for the test.
    test_client_id = 999999

    response = api_client.post(
        "/predict",
        params={"client_id": test_client_id},
        json=records_payload
    )
    
    # Check status code
    assert response.status_code == status.HTTP_200_OK, f"API returned {response.status_code}: {response.text}"
    
    # Check response body structure and types for a single prediction result
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    
    # Check response values
    assert 0.0 <= data["probability"] <= 1.0