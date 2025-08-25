import streamlit as st
import requests
import pandas as pd
import json
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Scoring Prediction",
    page_icon="💳",
    layout="centered"
)

# --- API Configuration ---
# Get the API URL from the environment variable set by Cloud Run
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") 
PREDICT_ENDPOINT = f"{API_URL}/predict"

# --- Expected Features (from the error message) ---
EXPECTED_FEATURES = [
    "AMT_ANNUITY", "ANNUITY_INCOME_PERC", "APPROVED_APP_CREDIT_PERC_MAX",
    "BURO_AMT_CREDIT_SUM_DEBT_SUM", "BURO_AMT_CREDIT_SUM_MEDIAN",
    "BURO_DAYS_CREDIT_MAX", "CODE_GENDER", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "DAYS_REGISTRATION", "EXT_SOURCE_2",
    "FLAG_DOCUMENT_3", "INCOME_PER_PERSON", "INSTAL_AMT_PAYMENT_MEDIAN",
    "INSTAL_AMT_PAYMENT_MIN", "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
    "INSTAL_DAYS_ENTRY_PAYMENT_MEDIAN", "INSTAL_DBD_MEAN",
    "NAME_EDUCATION_TYPE_Secondary / secondary special", "NAME_INCOME_TYPE_Working",
    "OCCUPATION_TYPE_Laborers", "PAYMENT_RATE", "PREV_AMT_ANNUITY_MIN",
    "PREV_APP_CREDIT_PERC_MEDIAN", "PREV_CNT_PAYMENT_MEAN",
    "PREV_CODE_REJECT_REASON_LIMIT_MEAN", "PREV_CODE_REJECT_REASON_SCOFR_MEAN",
    "PREV_DAYS_DECISION_MAX", "PREV_HOUR_APPR_PROCESS_START_MEAN",
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN", "PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN",
    "PREV_NAME_PRODUCT_TYPE_walk-in_MEAN", "PREV_NAME_YIELD_GROUP_low_normal_MEAN",
    "PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN", "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT_W_CITY", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
    "BURO_DAYS_CREDIT_ENDDATE_MEAN_pow0_5", "BURO_DAYS_CREDIT_MEAN_pow0_5",
    "DAYS_BIRTH_pow2", "SK_ID_CURR_pow2"
]

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

# Download template button
template_data = {feature: 0.0 for feature in EXPECTED_FEATURES}
st.sidebar.download_button(
    "Download Template JSON",
    json.dumps(template_data, indent=2),
    "client_data_template.json",
    "application/json"
)

# "Get Prediction" button
predict_button = st.sidebar.button("Get Prediction", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "This is a demo application. The prediction is based on a machine learning model "
    "and should not be used for real financial decisions without further validation."
)

# Display expected features
with st.expander("Show Expected Features"):
    st.write(f"The model expects {len(EXPECTED_FEATURES)} features:")
    for i, feature in enumerate(EXPECTED_FEATURES, 1):
        st.write(f"{i}. {feature}")

# Function to convert all values to float
def convert_values_to_float(data_dict):
    """Convert all values in a dictionary to float to match model expectations."""
    converted = {}
    for key, value in data_dict.items():
        try:
            converted[key] = float(value)
        except (ValueError, TypeError):
            # If conversion fails, use 0.0 as default
            converted[key] = 0.0
    return converted

# --- Main Panel for Results ---
if predict_button and uploaded_file is not None:
    if not client_id:
        st.error("Please enter a Client ID.")
    else:
        with st.spinner("Analyzing client data..."):
            try:
                # 1. Load and parse the uploaded JSON file
                client_data = json.load(uploaded_file)
                
                # 2. Convert all values to float to match model expectations
                client_data = convert_values_to_float(client_data)
                
                # 3. Check for missing features and set defaults
                for feature in EXPECTED_FEATURES:
                    if feature not in client_data:
                        client_data[feature] = 0.0
                        st.warning(f"Missing feature '{feature}' set to 0.0")
                
                # 4. Prepare the payload in the format the API expects
                api_payload = {"features": client_data}

                # 5. Send request to the FastAPI endpoint
                response = requests.post(
                    f"{PREDICT_ENDPOINT}?client_id={client_id}", 
                    json=api_payload
                )
                
                # 6. Handle the response
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Prediction successful for Client ID: **{result['client_id']}**")
                    
                    # Display metrics
                    prob = result['probability']
                    prediction = result['prediction']
                    threshold = result['threshold']
                    
                    status = "High Risk (Likely to Default)" if prediction == 1 else "Low Risk (Unlikely to Default)"
                    status_color = "red" if prediction == 1 else "green"
                    
                    st.markdown(f"### Prediction Status: :{status_color}[{status}]")
                    
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
                    try:
                        error_detail = response.json()
                        st.error(f"Error Details: {error_detail.get('detail', 'Unknown error')}")
                    except:
                        st.error(f"Error: {response.text}")

            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid JSON file.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif predict_button:
    st.warning("Please upload a client data file.")

# Add health check
try:
    health_response = requests.get(f"{API_URL}/")
    if health_response.status_code == 200:
        st.sidebar.success("✅ API is running and connected")
    else:
        st.sidebar.error("❌ API is not responding")
except:
    st.sidebar.error("❌ Cannot connect to API. Make sure the server is running.")

# Add sample data generator
if st.sidebar.checkbox("Generate Sample Data"):
    sample_data = {}
    for feature in EXPECTED_FEATURES:
        if "DAYS" in feature:
            # For days features, use negative values (days before current date)
            sample_data[feature] = float(np.random.uniform(-20000, 0))
        elif "AMT" in feature or "CREDIT" in feature or "ANNUITY" in feature:
            sample_data[feature] = float(np.random.uniform(0, 1000000))
        elif "PERC" in feature or "RATE" in feature:
            sample_data[feature] = float(np.random.uniform(0, 1))
        elif "FLAG" in feature or "CODE" in feature:
            sample_data[feature] = float(np.random.choice([0, 1]))
        elif "pow" in feature:
            sample_data[feature] = float(np.random.uniform(0, 100))
        else:
            sample_data[feature] = float(np.random.uniform(0, 10))
    
    st.sidebar.download_button(
        "Download Sample Data",
        json.dumps(sample_data, indent=2),
        "sample_data.json",
        "application/json"
    )