# ai_input_1.py

"""
This script is designed to train a machine learning model for credit scoring using a Random Forest classifier.
It includes data preprocessing, feature selection, hyperparameter tuning, and model evaluation.
The model is logged and registered with MLflow for tracking and versioning, including its signature.
The script also includes custom scoring metrics and handles missing values and outliers in the dataset.
It uses a pipeline to streamline the process and allows for easy adjustments to the model and preprocessing steps.
"""
import time
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    make_scorer,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import cross_val_score
import optuna
import matplotlib
import tempfile
import shutil
import atexit
from mlflow.models.signature import infer_signature

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "Credit_Scoring_Pipeline"
REGISTERED_MODEL_NAME = "CreditScoringRF"
DATA_PATH = "/app/data_source/df_global.parquet"


ARTIFACTS_DIR = "/mlflow_artifact"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

evaluation_dir = os.path.join(ARTIFACTS_DIR, "evaluation_artifacts")
os.makedirs(evaluation_dir, exist_ok=True)

# --- Global Configuration Dictionary ---
config = {
    "data_preprocessing": {
        "completeness_threshold": 85,
        "impute_method": "median",
        "outlier_percent": 0.2,
    },
    "feature_selection": {
        "n_features": 45,
    },
    "model_params": {
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "optuna": {
        "n_trials": 10
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "scaler": {
        "method": "RobustScaler",  # Options: 'StandardScaler', 'MinMaxScaler', 'RobustScaler'
    },
}

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# --- Data Processing Functions ---
def preprocess_dataframe(
    df, target_col, completeness=85, impute="median", verbose=True
):
    df_processed = df.copy()
    col_completeness = (1 - df_processed.isnull().sum() / len(df_processed)) * 100
    row_completeness = (
        1 - df_processed.isnull().sum(axis=1) / df_processed.shape[1]
    ) * 100
    cols_to_drop = col_completeness[col_completeness < completeness].index.tolist()
    if cols_to_drop:
        df_processed.drop(columns=cols_to_drop, inplace=True)
    rows_to_drop = df_processed[row_completeness < completeness * 0.5].index.tolist()
    if rows_to_drop:
        df_processed.drop(index=rows_to_drop, inplace=True)
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if impute == "median":
        impute_value = df_processed[numerical_cols].median()
    elif impute == "mean":
        impute_value = df_processed[numerical_cols].mean()
    else:  # zero
        impute_value = 0
    df_processed[numerical_cols] = df_processed[numerical_cols].fillna(impute_value)
    df_processed.dropna(subset=[target_col], inplace=True)
    df_processed[target_col] = df_processed[target_col].astype(int)
    return df_processed


def remove_percent_outliers(df, percent, target_col):
    df_cleaned = df.copy()
    features_to_clean = [
        col for col in df.columns if col not in [target_col, "SK_ID_CURR", "index"]
    ]
    indices_to_drop = set()
    lower_quantile = percent / 100
    upper_quantile = 1 - (percent / 100)
    for feat in features_to_clean:
        if df_cleaned[feat].dtype in [np.float64, np.int64]:
            lower_val = df_cleaned[feat].quantile(lower_quantile)
            upper_val = df_cleaned[feat].quantile(upper_quantile)
            outlier_indices = df_cleaned[
                (df_cleaned[feat] < lower_val) | (df_cleaned[feat] > upper_val)
            ].index
            indices_to_drop.update(outlier_indices)
    df_cleaned.drop(index=list(indices_to_drop), inplace=True)
    return df_cleaned


# --- Custom Scorer ---
def custom_confusion_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return -100000
    tn, fp, fn, tp = cm.ravel()
    return (tp * 2) - (fp * 1) - (fn * 6)


custom_scorer = make_scorer(custom_confusion_score, greater_is_better=True)


def main():
    """Main function to run the training pipeline."""
    # mlflow.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    print("Loading and preparing data...")
    df_global = pd.read_parquet(DATA_PATH)
    df_processed = preprocess_dataframe(
        df_global,
        target_col="TARGET",
        completeness=config["data_preprocessing"]["completeness_threshold"],
        impute=config["data_preprocessing"]["impute_method"],
    )
    df_cleaned = remove_percent_outliers(
        df_processed,
        percent=config["data_preprocessing"]["outlier_percent"],
        target_col="TARGET",
    )
    X = df_cleaned.drop(["TARGET", "SK_ID_CURR", "index"], axis=1, errors="ignore")
    y = df_cleaned["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["train_test_split"]["test_size"],
        random_state=config["train_test_split"]["random_state"],
        stratify=y,
    )

    # --- Save Processed Data Locally ---
    # This data will be logged as an artifact within the MLflow run.
    processed_data_dir = os.path.join(ARTIFACTS_DIR, "processed_data_temp")
    os.makedirs(processed_data_dir, exist_ok=True)
    X_train.to_parquet(os.path.join(processed_data_dir, "X_train.parquet"))
    X_test.to_parquet(os.path.join(processed_data_dir, "X_test.parquet"))
    y_train.to_frame().to_parquet(os.path.join(processed_data_dir, "y_train.parquet"))
    y_test.to_frame().to_parquet(os.path.join(processed_data_dir, "y_test.parquet"))
    print(
        f"Processed data temporarily saved to {processed_data_dir} for artifact logging."
    )

    if config["scaler"]["method"] == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif config["scaler"]["method"] == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # --- MLflow Parent Run ---
    with mlflow.start_run(run_name="Parent_Run_Manual_Logging") as parent_run:
        run_id = parent_run.info.run_id
        print(f"Starting Parent MLflow Run: {run_id}")
        mlflow.log_params(config)

        # --- Log Processed Data Artifacts ---
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data artifacts to MLflow.")

        def objective(trial):
            # --- Nested MLflow Run for each Optuna trial ---
            with mlflow.start_run(
                run_name=f"Trial_{{trial.number}}", nested=True
            ) as child_run:
                params = {
                    "clf__n_estimators": trial.suggest_int(
                        "clf__n_estimators", 100, 300
                    ),
                    "clf__max_depth": trial.suggest_int("clf__max_depth", 5, 15),
                    "clf__min_samples_split": trial.suggest_int(
                        "clf__min_samples_split", 2, 10
                    ),
                    "clf__min_samples_leaf": trial.suggest_int(
                        "clf__min_samples_leaf", 1, 5
                    ),
                }
                mlflow.log_params(params)

                pipeline = Pipeline(
                    [
                        ("selector", "passthrough"),  # No feature selection
                        ("scaler", scaler),
                        ("clf", RandomForestClassifier(**config["model_params"])),
                    ]
                )
                pipeline.set_params(**params)

                # --- Validation Step ---
                scores = cross_val_score(
                    pipeline, X_train, y_train, cv=3, scoring=custom_scorer, n_jobs=-1
                )
                validation_score = scores.mean()

                # --- Log Validation Metric ---
                mlflow.log_metric("validation_custom_score", validation_score)

                return validation_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["optuna"]["n_trials"])

        best_params = study.best_params
        print(f"Optuna best params: {best_params}")
        mlflow.log_params({"best_hyperparameters": best_params})

        # --- Train Final Model on Full Training Data ---
        final_pipeline = Pipeline(
            [
                ("selector", "passthrough"),  # No feature selection
                ("scaler", scaler),
                ("clf", RandomForestClassifier(**config["model_params"])),
            ]
        )
        final_pipeline.set_params(**best_params)
        final_model = final_pipeline.fit(X_train, y_train)

        # --- Final Evaluation on Test Set ---
        print("Evaluating final model on the test set...")
        y_proba = final_model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.1, 0.95, 0.01)
        best_threshold, best_score = 0.5, -np.inf
        for t in thresholds:
            score = custom_confusion_score(y_test, (y_proba >= t).astype(int))
            if score > best_score:
                best_score, best_threshold = score, t
        y_pred_final = (y_proba >= best_threshold).astype(int)

        print(f"Best threshold on test set: {best_threshold:.4f}")

        # --- Log Test Metrics ---
        test_metrics = {
            "test_custom_score": best_score,
            "test_f1_score": f1_score(y_test, y_pred_final),
            "test_accuracy": accuracy_score(y_test, y_pred_final),
            "test_precision": precision_score(y_test, y_pred_final),
            "test_recall": recall_score(y_test, y_pred_final),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_optimal_threshold": best_threshold,
        }
        mlflow.log_metrics(test_metrics)
        print(f"Logged test metrics: {test_metrics}")

        # --- Create and Save All Artifacts ---
        cm = confusion_matrix(y_test, y_pred_final)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Test Set)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        # Save all evaluation artifacts in the evaluation_dir
        cm_path = os.path.join(evaluation_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close("all")

        features = X_train.columns.tolist()
        features_path = os.path.join(evaluation_dir, "model_features.json")
        with open(features_path, "w") as f:
            json.dump(features, f)

        threshold_path = os.path.join(evaluation_dir, "optimal_threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(best_threshold))

        mlflow.log_artifacts(evaluation_dir, artifact_path="evaluation_artifacts")

        # --- Log Model with Signature and Register ---
        print("Logging and registering model with signature to MLflow...")
        signature = infer_signature(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="credit_scoring_model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Model registered as '{REGISTERED_MODEL_NAME}'.")

        # --- Transition Model to Production ---
        try:
            latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=latest_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Successfully transitioned model version {latest_version.version} to Production.")
        except Exception as e:
            print(f"Error transitioning model to Production: {e}")

    print("\nPipeline training, logging, and registration complete.")


if __name__ == "__main__":

    temp_dir = tempfile.mkdtemp()
    joblib.Memory(location=temp_dir, verbose=0)
    main()

    def cleanup():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    atexit.register(cleanup)
