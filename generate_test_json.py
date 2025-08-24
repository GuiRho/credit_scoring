import pandas as pd
import json

# --- IMPORTANT ---
# Update this path to point to the EXACT test data your best model was evaluated on.
# You can find the path in the MLflow UI under the run's parameters ("input_dir").

TEST_DATA_PATH = "C:/Users/gui/Documents/OpenClassrooms/Projet 7/cache/processed_s80_c60_robust/test_processed.parquet"

MODEL_EXPECTED_FEATURES = [
    "AMT_ANNUITY",
    "ANNUITY_INCOME_PERC",
    "APPROVED_APP_CREDIT_PERC_MAX",
    "BURO_AMT_CREDIT_SUM_DEBT_SUM",
    "BURO_AMT_CREDIT_SUM_MEDIAN",
    "BURO_DAYS_CREDIT_MAX",
    "CODE_GENDER",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_REGISTRATION",
    "EXT_SOURCE_2",
    "FLAG_DOCUMENT_3",
    "INCOME_PER_PERSON",
    "INSTAL_AMT_PAYMENT_MEDIAN",
    "INSTAL_AMT_PAYMENT_MIN",
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
    "INSTAL_DAYS_ENTRY_PAYMENT_MEDIAN",
    "INSTAL_DBD_MEAN",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_INCOME_TYPE_Working",
    "OCCUPATION_TYPE_Laborers",
    "PAYMENT_RATE",
    "PREV_AMT_ANNUITY_MIN",
    "PREV_APP_CREDIT_PERC_MEDIAN",
    "PREV_CNT_PAYMENT_MEAN",
    "PREV_CODE_REJECT_REASON_LIMIT_MEAN",
    "PREV_CODE_REJECT_REASON_SCOFR_MEAN",
    "PREV_DAYS_DECISION_MAX",
    "PREV_HOUR_APPR_PROCESS_START_MEAN",
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN",
    "PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN",
    "PREV_NAME_PRODUCT_TYPE_walk-in_MEAN",
    "PREV_NAME_YIELD_GROUP_low_normal_MEAN",
    "PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "BURO_DAYS_CREDIT_ENDDATE_MEAN_pow0_5",
    "BURO_DAYS_CREDIT_MEAN_pow0_5",
    "DAYS_BIRTH_pow2",
    "SK_ID_CURR_pow2"
]

INT_FEATURES = [
    "CODE_GENDER",
    "DAYS_ID_PUBLISH",
    "FLAG_DOCUMENT_3",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_INCOME_TYPE_Working",
    "OCCUPATION_TYPE_Laborers",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "DAYS_BIRTH_pow2",
    "SK_ID_CURR_pow2"
]

# Load the data
try:
    df_test = pd.read_parquet(TEST_DATA_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {TEST_DATA_PATH}")
    print("Please update the path to your 'test_processed.parquet' file.")
    exit()

# Get the first row of data (features only), selecting only the expected features
sample_client = df_test[MODEL_EXPECTED_FEATURES].iloc[0]

# Convert specified features to int
for col in INT_FEATURES:
    if col in sample_client:
        sample_client[col] = int(sample_client[col])

# Convert it to a dictionary suitable for the API
# The 'features' key is required by our Pydantic model
api_payload = {
    "features": sample_client.to_dict()
}

# Print the JSON to the console
print(json.dumps(api_payload, indent=2))

# Optional: save to a file
with open("sample_payload.json", "w") as f:
    json.dump(api_payload, f, indent=2)
print("\nPayload printed above and saved to sample_payload.json")