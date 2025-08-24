import streamlit as st
import requests
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Scoring Prediction",
    page_icon="💳",
    layout="centered"
)

# --- API Configuration ---
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Components ---
st.title("💳 Credit Scoring & Default Prediction")
st.write(
    "Upload a client's data in JSON format to receive a credit score prediction. "
    "The model will predict the probability of the client defaulting on their loan."
)

st.sidebar.header("Client Information")
client_id = st.sidebar.text_input("Enter Client ID", "SK_ID_12345")

# File uploader allows users to upload a single JSON file
uploaded_file = st.sidebar.file_uploader(
    "Upload Client Data (JSON)", 
    type=["json"]
)

# "Get Prediction" button
predict_button = st.sidebar.button("Get Prediction", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "This is a demo application. The prediction is based on a machine learning model "
    "and should not be used for real financial decisions without further validation."
)

# --- Main Panel for Results ---
if predict_button and uploaded_file is not None:
    if not client_id:
        st.error("Please enter a Client ID.")
    else:
        with st.spinner("Analyzing client data..."):
            try:
                # 1. Load and parse the uploaded JSON file
                client_data = json.load(uploaded_file)
                
                # The API expects the data under a 'features' key
                # We'll assume the uploaded JSON is the dictionary of features itself
                api_payload = {"features": client_data}

                # 2. Send request to the FastAPI endpoint
                response = requests.post(
                    f"{API_URL}?client_id={client_id}", 
                    json=api_payload
                )
                
                # 3. Handle the response
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Prediction successful for Client ID: **{result['client_id']}**")
                    
                    # Display metrics
                    prob = result['probability']
                    prediction = result['prediction']
                    threshold = result['threshold']
                    
                    status = "Likely to Default" if prediction == 1 else "Unlikely to Default"
                    
                    st.metric(
                        label="Prediction Status",
                        value=status
                    )
                    
                    # A visual gauge for the probability
                    st.progress(prob, text=f"Default Probability: {prob:.2%}")
                    
                    if prediction == 1:
                        st.warning(f"The model's predicted probability ({prob:.2%}) is above the decision threshold ({threshold:.2%}).")
                    else:
                        st.info(f"The model's predicted probability ({prob:.2%}) is below the decision threshold ({threshold:.2%}).")
                        
                    # Show the full JSON response in an expander
                    with st.expander("Show Full API Response"):
                        st.json(result)

                else:
                    st.error(f"API Error: Status Code {response.status_code}")
                    st.json(response.json())

            except Exception as e:
                st.error(f"An error occurred: {e}")

elif predict_button:
    st.warning("Please upload a client data file.")