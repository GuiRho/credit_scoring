import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

warnings.filterwarnings("ignore")

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
REGISTERED_MODEL_NAME = "CreditScoringRF"
ARTIFACTS_DIR = "C:/Users/gui/Documents/OpenClassrooms/Projet 7/mlflow_artifact"
EVIDENTLY_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "data_drift_report.html")

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_production_model_details(registered_model_name):
    """Retrieves the latest production model version and its run ID."""
    print(f"Fetching production model details for '{registered_model_name}'...")
    try:
        prod_versions = client.get_latest_versions(
            name=registered_model_name, stages=["Production"]
        )
        if not prod_versions:
            raise ValueError(
                f"No model version found for '{registered_model_name}' in stage 'Production'."
            )
        latest_prod_version = prod_versions[0]
        run_id = latest_prod_version.run_id
        model_uri = f"runs:/{run_id}/credit_scoring_model"
        print(
            f"Found production model version {latest_prod_version.version} from run_id: {run_id}"
        )
        return model_uri, run_id
    except Exception as e:
        print(f"Error fetching model details: {e}")
        return None, None


def load_data_from_run(run_id):
    """Loads data artifacts from a specific MLflow run."""
    print(f"Loading data artifacts from run_id: {run_id}")
    try:
        local_dir = client.download_artifacts(run_id, "processed_data")
        X_train = pd.read_parquet(os.path.join(local_dir, "X_train.parquet"))
        X_test = pd.read_parquet(os.path.join(local_dir, "X_test.parquet"))
        print("Successfully loaded X_train and X_test data.")
        return X_train, X_test
    except Exception as e:
        print(f"Error loading data artifacts: {e}")
        return None, None


def run_drift_analysis():
    """
    Main function to run the data drift analysis.
    - Retrieves the production model from MLflow.
    - Calculates feature importances.
    - Selects the top 20 features.
    - Runs an Evidently AI data drift report on these features.
    """
    model_uri, run_id = get_production_model_details(REGISTERED_MODEL_NAME)
    if not model_uri:
        print("Exiting analysis due to model loading error.")
        return

    # 1. Retrieve Model and Feature Names
    model = mlflow.sklearn.load_model(model_uri)
    print("Successfully loaded model pipeline.")

    print("Loading X_train to get feature list.")
    X_train_temp, _ = load_data_from_run(run_id)
    if X_train_temp is None:
        print("Could not load X_train to get feature names. Exiting.")
        return
    feature_names = X_train_temp.columns.tolist()


    # 2. Calculate Feature Importances (MDI)
    print("Calculating feature importances (MDI)...")
    rf_classifier = model.named_steps["clf"]
    importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    )

    # 3. Select Top 20 Features
    top_20_features = feature_importance_df.sort_values(
        by="importance", ascending=False
    ).head(20)
    top_20_feature_list = top_20_features["feature"].tolist()
    print("Identified Top 20 most important features:")
    print(top_20_features)

    # 4. Load Data for Drift Analysis
    X_train, X_test = load_data_from_run(run_id)
    if X_train is None or X_test is None:
        print("Exiting analysis due to data loading error.")
        return

    # 5. Run Evidently AI Report on Top 20 Features
    print(f"Running Evidently AI data drift report on {len(top_20_feature_list)} features...")

    # Define the column mapping for the report
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = top_20_feature_list

    # Create and run the report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=X_train[top_20_feature_list],
        current_data=X_test[top_20_feature_list],
        column_mapping=column_mapping,
    )

    # Save the report
    drift_report.save_html(EVIDENTLY_REPORT_PATH)
    print(f"Evidently AI report saved to: {EVIDENTLY_REPORT_PATH}")


if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    run_drift_analysis()