import os
import json
import argparse
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------
# Helper Functions
# -------------------

def setup_mlflow(cache_dir: str, experiment_name: str):
    """Sets up MLflow tracking URI and experiment."""
    try:
        # Prefer a file-based URI for consistency
        mlruns_dir = os.path.join(os.path.abspath(cache_dir), "mlruns")
        mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
    except Exception:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(cache_dir)}")
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment set to: '{experiment_name}'")

def get_resampling_strategy(technique: str, target_balance: float, initial_balance: float, random_state: int):
    """
    Returns the configured imblearn resampling object based on the desired technique.
    """
    if technique == "oversample":
        # We can only oversample if the target is greater than the initial balance
        if target_balance <= initial_balance:
            print(f"Skipping oversample: target balance {target_balance:.2f} is not > initial balance {initial_balance:.2f}")
            return None
        return SMOTE(sampling_strategy=target_balance, random_state=random_state)

    elif technique == "undersample":
        # For RandomUnderSampler, sampling_strategy is the ratio of minority to majority
        # We must convert our target proportion (e.g., 0.4) to this ratio.
        # If target is 40% minority, then 60% majority. Ratio = 40/60.
        if target_balance == 1.0: return None # Avoid division by zero
        sampling_ratio = target_balance / (1 - target_balance)
        return RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state)

    elif technique == "hybrid":
        if target_balance <= initial_balance:
            print(f"Skipping hybrid: target balance {target_balance:.2f} is not > initial balance {initial_balance:.2f}")
            return None
        
        # 1. Oversample halfway to the target
        midpoint_balance = (initial_balance + target_balance) / 2
        over = SMOTE(sampling_strategy=midpoint_balance, random_state=random_state)
        
        # 2. Undersample the rest of the way to the final target ratio
        sampling_ratio = target_balance / (1 - target_balance)
        under = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state)
        
        return ImbPipeline(steps=[('over', over), ('under', under)])
        
    else:
        raise ValueError(f"Unknown resampling technique: {technique}")

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes a numeric dataframe by handling inf, large values, and NaNs."""
    df_sanitized = df.select_dtypes(include=np.number).copy()
    df_sanitized.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Simple imputation with medians
    if df_sanitized.isna().sum().sum() > 0:
        medians = df_sanitized.median()
        df_sanitized.fillna(medians, inplace=True)
        
    return df_sanitized

# -------------------
# Main Evaluation Logic
# -------------------

def evaluate_balancing_strategies(
    input_dir: str, 
    target_col: str = "TARGET", 
    cache_dir: str = "cache",
    random_state: int = 42
):
    """
    Loads pre-split train and test data, then iterates through various balancing strategies,
    training a model for each and evaluating on the untouched test set.
    """
    print(f"Loading data from input directory: {input_dir}")
    train_path = os.path.join(input_dir, "train_processed.parquet")
    test_path = os.path.join(input_dir, "test_processed.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train/test parquet files not found in directory: {input_dir}")

    df_train = pd.read_parquet(train_path, engine="pyarrow")
    df_test = pd.read_parquet(test_path, engine="pyarrow")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Sanitize dataframes to prevent issues with resampling and modeling
    X_train = sanitize_dataframe(X_train)
    X_test = sanitize_dataframe(X_test)
    
    initial_balance = y_train.value_counts(normalize=True).get(1, 0)
    print(f"Initial minority class balance in training set: {initial_balance:.3f}")
    print(f"Test set shape: {X_test.shape}. Test set will remain untouched for evaluation.")

    # Define the parameter grid for the experiment
    target_balances = np.arange(0.10, 0.51, 0.05)
    techniques = ["undersample", "oversample", "hybrid"]

    for target_balance in target_balances:
        for technique in techniques:
            run_name = f"balance_{Path(input_dir).name}_{target_balance:.2f}_{technique}"
            print(f"\n=================================================")
            print(f"Starting MLflow run: {run_name}")
            print(f"=================================================")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "input_dir": input_dir,
                    "technique": technique,
                    "target_balance": target_balance,
                    "initial_train_balance": initial_balance
                })
                
                resampler = get_resampling_strategy(technique, target_balance, initial_balance, random_state)
                if resampler is None:
                    print("Strategy not applicable, skipping run.")
                    continue
                
                try:
                    print(f"Applying '{technique}' resampling...")
                    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                    print(f"Resampled training set shape: {X_resampled.shape}")
                except Exception as e:
                    print(f"Error during resampling for {run_name}: {e}")
                    mlflow.log_param("error", str(e))
                    continue

                final_resampled_balance = y_resampled.value_counts(normalize=True).get(1, 0)
                mlflow.log_metrics({
                    "resampled_train_rows": X_resampled.shape[0],
                    "final_resampled_balance": final_resampled_balance
                })

                print("Training Logistic Regression model...")
                model_pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
                ])
                model_pipeline.fit(X_resampled, y_resampled)
                
                print("Evaluating model on the untouched test set...")
                y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                print(f"Test ROC AUC Score: {test_roc_auc:.4f}")
                mlflow.log_metric("test_roc_auc", test_roc_auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple data balancing strategies using pre-split data.")
    parser.add_argument("--input-dir", required=True, help="Path to the directory containing train_processed.parquet and test_processed.parquet.")
    parser.add_argument("--target", default="TARGET", help="Name of the target column.")
    parser.add_argument("--cache-dir", default="cache", help="Directory to store MLflow runs and other cache files.")
    
    args = parser.parse_args()
    
    setup_mlflow(args.cache_dir, "Balancing Strategies Comparison")
    
    evaluate_balancing_strategies(
        input_dir=args.input_dir,
        target_col=args.target,
        cache_dir=args.cache_dir
    )

    print("\nExperiment complete. Check the MLflow UI for results.")