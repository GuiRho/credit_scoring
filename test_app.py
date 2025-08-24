import streamlit as st
import requests
import json
import os

# Path to the generated sample_payload.json
PAYLOAD_PATH = "sample_payload.json"
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(layout="wide", page_title="Credit Scoring API Tester")
st.title("Credit Scoring API Tester")

# Load the sample payload
@st.cache_data
def load_payload(path):
    if not os.path.exists(path):
        st.error(f"Payload file not found at: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

# Display the payload
raw_payload = load_payload(PAYLOAD_PATH)

if raw_payload:
    st.subheader("Generated JSON Payload")
    st.json(raw_payload)

    st.subheader("API Interaction")
    client_id = st.text_input("Enter Client ID (e.g., test_client)", "test_client")

    if st.button("Send Prediction Request"):
        if not client_id:
            st.warning("Please enter a Client ID.")
        else:
            try:
                # Make the POST request
                headers = {'Content-Type': 'application/json'}
                response = requests.post(f"{API_URL}?client_id={client_id}", json=raw_payload, headers=headers)

                st.subheader("API Response")
                if response.status_code == 200:
                    st.success("Request Successful!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.status_code} - {response.reason}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Please ensure the FastAPI server is running at http://127.0.0.1:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.warning("Please run `python generate_test_json.py` to create `sample_payload.json` first.")
