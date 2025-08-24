import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def clean_and_impute_data(
    df,
    target_col='TARGET',
    completeness=85,
    impute='median',
    verbose=True,
    cv_threshold: float = 0.01
):
    """
    Clean and impute data, dropping columns with low completeness or low coefficient of variation (CV).
    """
    df_processed = df.copy()
    initial_cols = df_processed.shape[1]
    initial_rows = df_processed.shape[0]

    # Drop columns based on completeness
    col_completeness = (1 - df_processed.isnull().sum() / len(df_processed)) * 100
    cols_to_drop_completeness = col_completeness[col_completeness < completeness].index.tolist()
    if target_col in cols_to_drop_completeness:
        cols_to_drop_completeness.remove(target_col)
    if cols_to_drop_completeness:
        df_processed.drop(columns=cols_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(cols_to_drop_completeness)} columns due to completeness < {completeness}%")

    # Drop rows based on completeness
    row_completeness = (1 - df_processed.isnull().sum(axis=1) / df_processed.shape[1]) * 100
    rows_to_drop_completeness = df_processed[row_completeness < completeness*0.5].index.tolist()
    if rows_to_drop_completeness:
        df_processed.drop(index=rows_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(rows_to_drop_completeness)} rows due to completeness < {completeness*0.5}%")

    # Impute missing values
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if impute == 'median':
        impute_value = df_processed[numerical_cols].median()
    elif impute == 'mean':
        impute_value = df_processed[numerical_cols].mean()
    elif impute == 'zero':
        impute_value = 0
    else:
        raise ValueError(f"Unknown impute method: {impute}")
    df_processed[numerical_cols] = df_processed[numerical_cols].fillna(impute_value)

    # Calculate CV for numerical columns (excluding target)
    if cv_threshold is not None and cv_threshold > 0:
        numerical_cols_after = df_processed.select_dtypes(include=np.number).columns.tolist()
        if target_col in numerical_cols_after:
            numerical_cols_after.remove(target_col)
        if numerical_cols_after:
            means = df_processed[numerical_cols_after].mean().abs()
            stds = df_processed[numerical_cols_after].std()
            cvs = stds.divide(means).replace([np.inf, -np.inf], 0)
            cvs = cvs.fillna(0)
            cols_to_drop_cv = cvs[cvs <= cv_threshold].index.tolist()
            if cols_to_drop_cv:
                df_processed.drop(columns=cols_to_drop_cv, inplace=True)
                if verbose:
                    print(f"Dropped {len(cols_to_drop_cv)} numerical columns with CV <= {cv_threshold}")

    # Handle target column
    if target_col in df_processed.columns:
        df_processed.dropna(subset=[target_col], inplace=True)
        df_processed[target_col] = df_processed[target_col].astype(int)

    if verbose:
        print(f"Original shape: ({initial_rows}, {initial_cols})")
        print(f"Processed shape: {df_processed.shape}")

    return df_processed

def remove_percent_outliers_2sides(df, percent, target_col='TARGET'):
    df_cleaned = df.copy()
    
    features = df_cleaned.drop(columns=[target_col], errors='ignore')
    
    for feat in features.columns:
        features[feat] = pd.to_numeric(features[feat], errors='coerce')
    
    features = features.dropna()
    
    if features.empty:
        return df_cleaned.loc[features.index]
        
    all_outlier_indices = set()
    lower_quantile = percent / 100
    upper_quantile = (100 - percent) / 100
    
    numeric_features = features.select_dtypes(include=[np.number])
    for feat in numeric_features.columns:
        lower_val = numeric_features[feat].quantile(lower_quantile)
        upper_val = numeric_features[feat].quantile(upper_quantile)
        outlier_rows = numeric_features[(numeric_features[feat] < lower_val) | (numeric_features[feat] > upper_val)]
        all_outlier_indices.update(outlier_rows.index.tolist())
        
    print(f"Number of total outliers found = {len(all_outlier_indices)}")
    df_cleaned = df_cleaned.drop(index=list(all_outlier_indices))
    return df_cleaned

# -------------------
# ML Evaluation and Logging
# -------------------
def evaluate_and_log(df: pd.DataFrame, target_col: str):
    """Trains a logistic regression model and logs the ROC AUC score to MLflow."""
    print("\n--- Starting Model Evaluation ---")
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Skipping evaluation.")
        return

    if df[target_col].nunique() < 2:
        print(f"Target column '{target_col}' has only one class. Skipping evaluation.")
        mlflow.log_metric("roc_auc", 0.0) # Log 0 or skip
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.select_dtypes(include=np.number)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metrics({
        "final_row_count": df.shape[0],
        "final_feature_count": df.shape[1] - 1
    })
    print("--- Finished Model Evaluation ---\n")

# -------------------
# Config and Main
# -------------------

@dataclass
class Config:
    run_name: str
    input_parquet: str
    output_parquet: str
    completeness: int = 85
    impute: str = "median"
    percent_outliers: int = 1
    target_col: str = "TARGET"
    verbose: bool = True
    cv_threshold: float = 0.01

def main(config_path: str):
    with open(config_path, 'r') as f:
        configs_data = json.load(f)

    for config_dict in configs_data:
        cfg = Config(**config_dict)
        
        print(f"=================================================")
        print(f"Starting MLflow run: {cfg.run_name}")
        print(f"=================================================")
        
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(asdict(cfg))
            
            print("Loading data from:", cfg.input_parquet)
            df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
            
            df_cleaned = clean_and_impute_data(
                df, 
                target_col=cfg.target_col, 
                completeness=cfg.completeness, 
                impute=cfg.impute, 
                verbose=cfg.verbose, 
                cv_threshold=cfg.cv_threshold
            )
            
            df_processed = remove_percent_outliers_2sides(df_cleaned, percent=cfg.percent_outliers, target_col=cfg.target_col)
            
            print("Saving processed data to:", cfg.output_parquet)
            output_dir = os.path.dirname(cfg.output_parquet)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df_processed.to_parquet(cfg.output_parquet, engine="pyarrow")
            
            mlflow.log_artifact(cfg.output_parquet, "processed_data")
            
            mlflow.log_param("nrow", df_processed.shape[0])
            mlflow.log_param("ncol", df_processed.shape[1])

            evaluate_and_log(df_processed, target_col=cfg.target_col)


if __name__ == "__main__":

    # Set tracking URI to the desired cache directory
    # Note the use of file:/// and url-encoding for the path
    tracking_uri = "file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("preprocess")
    print(f"MLflow is configured to track experiments to: {mlflow.get_tracking_uri()}")

    # The rest of the script is for command-line argument parsing and execution.
    parser = argparse.ArgumentParser(description="Run preprocessing pipelines from a JSON config file and log to MLflow.")
    parser.add_argument("--config", required=True, dest="config_path", help="Path to the preprocess_config.json file")
    
    args = parser.parse_args()
    
    main(args.config_path)
