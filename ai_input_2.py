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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # NEW IMPORT
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Credit_Scoring_Pipeline"
REGISTERED_MODEL_NAME = "CreditScoringRF"
DATA_PATH = "C:/Users/gui/Documents/OpenClassrooms/Projet 7/Enonce/df_global.parquet"


ARTIFACTS_DIR = "C:/Users/gui/Documents/OpenClassrooms/Projet 7/mlflow_artifact"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

evaluation_dir = os.path.join(ARTIFACTS_DIR, "evaluation_artifacts")
os.makedirs(evaluation_dir, exist_ok=True)

# --- Global Configuration Dictionary ---
config = {
    "data_preprocessing": {
        "completeness_threshold": 85,
        "impute_method": "median",
        "outlier_percent": 0.5,
    },
    "feature_selection": {
        "n_features": 25,
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
    # Ensure no unnamed index becomes a column - this should handle 'index' too
    df_processed = df_processed.loc[:, ~df_processed.columns.str.contains('^Unnamed')]
    if 'index' in df_processed.columns: # Explicitly drop if 'index' still exists
        df_processed = df_processed.drop(columns=['index'])
    
    col_completeness = (1 - df_processed.isnull().sum() / len(df_processed)) * 100
    rows_to_drop_by_completeness = (
        (1 - df_processed.isnull().sum(axis=1) / df_processed.shape[1]) * 100 < completeness * 0.5
    )
    
    cols_to_drop = col_completeness[col_completeness < completeness].index.tolist()
    if cols_to_drop:
        if verbose: print(f"Dropping columns due to low completeness: {cols_to_drop}")
        df_processed.drop(columns=cols_to_drop, inplace=True)
    
    if rows_to_drop_by_completeness.any():
        if verbose: print(f"Dropping {rows_to_drop_by_completeness.sum()} rows due to low completeness.")
        df_processed = df_processed[~rows_to_drop_by_completeness]
    
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    if impute == "median":
        impute_values = df_processed[numerical_cols].median()
    elif impute == "mean":
        impute_values = df_processed[numerical_cols].mean()
    else:  # zero
        impute_values = 0
    
    df_processed[numerical_cols] = df_processed[numerical_cols].fillna(impute_values)
    
    df_processed.dropna(subset=[target_col], inplace=True) # Ensure target is not null
    df_processed[target_col] = df_processed[target_col].astype(int)
    
    return df_processed


def remove_percent_outliers(df, percent, target_col):
    df_cleaned = df.copy()
    # Features to clean should be numerical. Categorical features are handled by OHE.
    features_to_clean = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if target_col in features_to_clean:
        features_to_clean.remove(target_col)
    if "SK_ID_CURR" in features_to_clean:
        features_to_clean.remove("SK_ID_CURR")

    indices_to_drop = set()
    lower_quantile = percent / 100
    upper_quantile = 1 - (percent / 100)
    for feat in features_to_clean:
        # Check if the feature is not constant (std > 0) to avoid errors with quantile on constant columns
        if df_cleaned[feat].std() > 1e-9: # Small epsilon to account for floating point
            lower_val = df_cleaned[feat].quantile(lower_quantile)
            upper_val = df_cleaned[feat].quantile(upper_quantile)
            outlier_indices = df_cleaned[
                (df_cleaned[feat] < lower_val) | (df_cleaned[feat] > upper_val)
            ].index
            indices_to_drop.update(outlier_indices)
    
    if indices_to_drop:
        df_cleaned.drop(index=list(indices_to_drop), inplace=True)
        print(f"Removed {len(indices_to_drop)} rows due to outliers.")
    else:
        print("No outliers found to remove.")
    return df_cleaned


# --- Custom Scorer ---
def custom_confusion_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        # This case implies an issue with predictions or true labels,
        # or that only one class was predicted/present.
        # Returning a very low score to discourage this.
        return -100000
    tn, fp, fn, tp = cm.ravel()
    # Assign higher penalty to False Negatives (missing a default)
    return (tp * 2) - (fp * 1) - (fn * 6)


custom_scorer = make_scorer(custom_confusion_score, greater_is_better=True)


def main():
    """Main function to run the training pipeline."""
    print("Loading and preparing data...")
    df_global = pd.read_parquet(DATA_PATH)
    
    # Ensure no default index is treated as a feature by explicitly dropping it if it exists
    # And resetting index to a clean 0-based index without making a column from the old index
    if 'index' in df_global.columns:
        df_global = df_global.drop(columns=['index'])
    df_global = df_global.reset_index(drop=True) 
    
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
    
    # Separate features (X) and target (y)
    # Ensure 'SK_ID_CURR' is dropped for the model
    X = df_cleaned.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
    y = df_cleaned["TARGET"]

    # Identify numerical and categorical columns for preprocessing
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    
    print(f"Identified {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features.")
    print(f"Categorical features: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["train_test_split"]["test_size"],
        random_state=config["train_test_split"]["random_state"],
        stratify=y,
    )

    # --- Save Processed Data Locally ---
    processed_data_dir = os.path.join(ARTIFACTS_DIR, "processed_data_temp")
    os.makedirs(processed_data_dir, exist_ok=True)
    X_train.to_parquet(os.path.join(processed_data_dir, "X_train.parquet"))
    X_test.to_parquet(os.path.join(processed_data_dir, "X_test.parquet"))
    y_train.to_frame().to_parquet(os.path.join(processed_data_dir, "y_train.parquet"))
    y_test.to_frame().to_parquet(os.path.join(processed_data_dir, "y_test.parquet"))
    print(
        f"Processed data temporarily saved to {processed_data_dir} for artifact logging."
    )

    # Define the scaler based on configuration
    if config["scaler"]["method"] == "MinMaxScaler":
        scaler_instance = MinMaxScaler()
    elif config["scaler"]["method"] == "RobustScaler":
        scaler_instance = RobustScaler()
    else: # Default to StandardScaler
        scaler_instance = StandardScaler()

    # Create the preprocessor using ColumnTransformer
    # Categorical features are one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler_instance, numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough' # Keep any other columns not explicitly listed (e.g., if any were missed)
    )

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
                run_name=f"Trial_{trial.number}", nested=True
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

                # Update pipeline to include the preprocessor
                pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
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
                ("preprocessor", preprocessor),
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

        cm_path = os.path.join(evaluation_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close("all")

        # Log the ORIGINAL feature names for reference, as the signature will be inferred from raw X_train
        features_path = os.path.join(evaluation_dir, "model_features.json")
        with open(features_path, "w") as f:
            json.dump(X_train.columns.tolist(), f) # Log original feature names for clarity

        threshold_path = os.path.join(evaluation_dir, "optimal_threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(best_threshold))

        mlflow.log_artifacts(evaluation_dir, artifact_path="evaluation_artifacts")

        # --- Log Model with Signature and Register ---
        print("Logging and registering model with signature to MLflow...")
        # CRUCIAL: Infer signature from the ORIGINAL, RAW X_train.
        # The pipeline handles internal transformations.
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
            # Get the latest version by querying models in "None" or "Production" stage
            latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None", "Production"])
            if not latest_versions:
                 print(f"No versions found for {REGISTERED_MODEL_NAME} to transition. Skipping stage transition.")
            else:
                latest_version = latest_versions[0] # Take the first one if multiple
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
    main()

    def cleanup():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    atexit.register(cleanup)