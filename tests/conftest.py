# tests/conftest.py

import pytest
import pickle
import json
from fastapi.testclient import TestClient
import pandas as pd
import sys
import os

# Add the model_serving directory to the Python path
# This allows us to import the 'app' from 'main.py'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_serving')))

# It's a good practice to try importing the app here to fail early if the path is wrong
try:
    from docker_main import app
except ImportError as e:
    print(f"Error importing FastAPI app: {e}")
    print("Please ensure 'model_serving/docker_main.py' exists and contains a FastAPI instance named 'app'.")
    # We can't proceed without the app, so we'll set it to None and let tests that need it fail.
    app = None

@pytest.fixture(scope="session")
def model():
    """
    Fixture to load the production model once per test session.
    """
    model_path = os.path.join("production_model", "model.pkl")
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except FileNotFoundError:
        pytest.fail(f"Model file not found at: {model_path}. Please ensure the model artifact is in place.")
    except Exception as e:
        pytest.fail(f"Failed to load model from {model_path}: {e}")

@pytest.fixture(scope="session")
def api_client():
    """
    Fixture to create a test client for the model serving API.
    """
    if app is None:
        pytest.fail("FastAPI 'app' could not be imported. Cannot create an API test client.")
    
    client = TestClient(app)
    return client

@pytest.fixture(scope="session")
def valid_input_data():
    """
    Fixture to load the example input data for the model.
    This represents the raw data before being converted to a DataFrame.
    """
    example_path = os.path.join("production_model", "input_example.json")
    try:
        with open(example_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Example input file not found at: {example_path}.")
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {example_path}.")

@pytest.fixture(scope="session")
def valid_serving_payload():
    """
    Fixture to load the example payload for the API.
    """
    example_path = os.path.join("production_model", "serving_input_example.json")
    try:
        with open(example_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Serving example input file not found at: {example_path}.")
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {example_path}.")