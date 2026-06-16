# tests/test_model_scoring.py

import pytest
import pandas as pd
import numpy as np

def test_model_loading(model):
    """
    Test 1: Check if the model fixture loads correctly.
    """
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_prediction_output(model, valid_input_data):
    """
    Test 2: Check the type and shape of the model's output.
    """
    df = pd.DataFrame(valid_input_data['data'], columns=valid_input_data['columns'])
    
    # Test predict()
    prediction = model.predict(df)
    assert isinstance(prediction, np.ndarray)
    
    # FIX: The input data contains 5 samples, so the output shape should be (5,).
    assert prediction.shape == (5,), f"Prediction shape should be (5,) but was {prediction.shape}"
    
    # Test predict_proba()
    probabilities = model.predict_proba(df)
    assert isinstance(probabilities, np.ndarray)
    
    # FIX: The input data contains 5 samples, so the probability shape should be (5, 2).
    assert probabilities.shape == (5, 2), f"Probabilities shape should be (5, 2) but was {probabilities.shape}"
    
    # FIX: Check that probabilities for EACH prediction sum to 1.
    assert np.all(np.isclose(np.sum(probabilities, axis=1), 1.0)), "Probabilities for each row should sum to 1"

def test_model_consistency(model, valid_input_data):
    """
    Test 3: Ensure the model gives a consistent prediction for a known input.
    """
    df = pd.DataFrame(valid_input_data['data'], columns=valid_input_data['columns'])
    
    expected_probability_class_1 = 0.03918383094454616
    
    probabilities = model.predict_proba(df)
    
    # Check the consistency of the first prediction in the batch.
    actual_probability_class_1 = probabilities[0, 1]
    
    assert actual_probability_class_1 == pytest.approx(expected_probability_class_1, abs=0.001)