import streamlit as st
import pandas as pd
import requests
import json
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "CreditScoringRF"
MODEL_STAGE = "Production"


@st.cache_data
def get_model_features():
    """
    Gets the feature list and their types from the latest model in the specified stage.
    """
    st.write(f"Connecting to MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Connecting to MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    st.write(f"Searching for model '{REGISTERED_MODEL_NAME}' in stage '{MODEL_STAGE}'")
    print(f"Searching for model '{REGISTERED_MODEL_NAME}' in stage '{MODEL_STAGE}'")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        latest_versions = client.get_latest_versions(
            name=REGISTERED_MODEL_NAME, stages=[MODEL_STAGE]
        )

        if not latest_versions:
            st.error(
                f"ERROR: No model versions found for '{REGISTERED_MODEL_NAME}' in stage '{MODEL_STAGE}'."
            )
            return None

        latest_version_info = latest_versions[0]
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version_info.version}"
        st.write(f"Found model. Fetching signature from URI: {model_uri}")

        model_info = mlflow.models.get_model_info(model_uri)

        if model_info.signature and model_info.signature.inputs:
            # Return a dictionary of feature_name: type
            features_with_types = {input.name: input.type for input in model_info.signature.inputs}
            st.success(f"Successfully loaded {len(features_with_types)} features from the model.")
            return features_with_types
        else:
            st.error("Model signature not found.")
            return None

    except Exception as e:
        st.error(f"A critical error occurred while fetching model features: {e}")
        return None


def call_api(data):
    """
    Sends data to the FastAPI prediction API and returns the prediction.
    """
    payload = {"data": [data]}
    print(f"Sending request to API: {API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print(f"API response status: {response.status_code}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        try:
            error_detail = http_err.response.json()
            st.error(f"API Error Detail: {error_detail.get('detail', 'No detail provided.')}")
        except json.JSONDecodeError:
            st.error(f"Could not parse error response: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling the prediction API: {e}")
        print(f"Error calling the prediction API: {e}")
        return None


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Credit Scoring Prediction", layout="wide")
    st.title("Credit Scoring Prediction :bank:")

    MODEL_FEATURES_AND_TYPES = get_model_features()

    if MODEL_FEATURES_AND_TYPES is None:
        st.warning("Could not load model features. The application cannot proceed.")
        return

    st.markdown(
        f"This application uses the **{REGISTERED_MODEL_NAME}** model (stage: **{MODEL_STAGE}**) to predict credit default risk."
    )
    st.markdown("Please enter the client's data in the sidebar to get a prediction.")

    st.sidebar.header("Client Information")
    input_data = {}
    for feature, feature_type in MODEL_FEATURES_AND_TYPES.items():
        # Assign sensible default values for each type
        default_value = None
        if feature_type == "integer":
            default_value = 0
            value = st.sidebar.number_input(
                f"Enter value for {feature} (Type: {feature_type})",
                value=default_value,
                step=1,
                format="%d",
                key=f"input_{feature}" # Add unique key for each widget
            )
            input_data[feature] = int(value)
        elif feature_type == "double":
            default_value = 0.0
            value = st.sidebar.number_input(
                f"Enter value for {feature} (Type: {feature_type})",
                value=default_value,
                format="%.4f",
                key=f"input_{feature}"
            )
            input_data[feature] = float(value)
        elif feature_type == "boolean":
            default_value = False
            value = st.sidebar.checkbox(
                f"Is {feature.replace('FLAG_OWN_', '').replace('_', ' ').lower()} owned?",
                value=default_value,
                key=f"input_{feature}"
            )
            input_data[feature] = 1 if value else 0 # Convert boolean to 1 or 0
        elif feature == "CODE_GENDER":
             # Provide common options, assume 'M' is a safe default
             options = ['M', 'F', 'XNA']
             default_index = options.index('M') if 'M' in options else 0
             value = st.sidebar.selectbox(
                f"Select gender for {feature} (Type: {feature_type})",
                options=options,
                index=default_index,
                key=f"input_{feature}"
            )
             input_data[feature] = value
        elif feature_type == "string":
            default_value = "" # Default to empty string
            value = st.sidebar.text_input(
                f"Enter value for {feature} (Type: {feature_type})",
                value=default_value,
                key=f"input_{feature}"
            )
            input_data[feature] = value
        else: # Fallback for any other unexpected types
            st.sidebar.warning(f"Feature '{feature}' has unhandled type '{feature_type}'. Using text input.")
            default_value = ""
            value = st.sidebar.text_input(
                f"Enter value for {feature} (Type: {feature_type})",
                value=default_value,
                key=f"input_{feature}"
            )
            input_data[feature] = value

    # Debugging: Show the payload before sending
    with st.expander("Input Data Payload (for Debugging)"):
        st.json({"data": [input_data]})

    if st.sidebar.button("Get Prediction"):
        with st.spinner("Getting prediction..."):
            prediction_result = call_api(input_data)
            if prediction_result:
                try:
                    final_prediction = prediction_result.get("predictions")[0]

                    st.subheader("Risk Assessment")

                    if final_prediction == 1:
                        st.error("Conclusion: **High Risk**")
                        st.metric(
                            label="Model Verdict",
                            value="Credit Default",
                            delta="Action Required",
                            delta_color="inverse",
                        )
                    else:
                        st.success("Conclusion: **Low Risk**")
                        st.metric(
                            label="Model Verdict",
                            value="No Credit Default",
                            delta="Likely Approved",
                            delta_color="normal",
                        )

                except (IndexError, TypeError, KeyError) as e:
                    st.error("Error parsing the API response.")
                    st.error(f"Parsing Error: {e}")

                with st.expander("Show Raw API Response"):
                    st.json(prediction_result)

    st.sidebar.markdown("---")
    st.sidebar.info("This app provides real-time predictions from a deployed machine learning model.")


if __name__ == "__main__":
    main()