import os
import json
import mlflow
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def resample_for_balance(df: pd.DataFrame, target_col: str, positive_fraction: float, random_state: int = 42):
    """
    Return a resampled dataframe with desired positive_fraction (between 0 and 1).
    If positive_fraction equals current fraction, returns original.
    """
    df = df.copy()
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    if pos.empty or neg.empty:
        return df
    n_total = len(df)
    n_pos = int(round(positive_fraction * n_total))
    n_pos = max(1, n_pos)
    n_neg = n_total - n_pos
    # sample with replacement if needed
    pos_sample = pos.sample(n=n_pos, replace=(n_pos > len(pos)), random_state=random_state)
    neg_sample = neg.sample(n=n_neg, replace=(n_neg > len(neg)), random_state=random_state)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def evaluate_dataframe(path_parquet: str, target_col: str = "TARGET", test_size: float = 0.2, random_state: int = 42, cache_dir: str = "cache"):
    df = pd.read_parquet(path_parquet, engine='pyarrow')
    balances = {
        "initial": None,
        "25_75": 0.25,
        "50_50": 0.5
    }
    results = {}
    mlflow.set_experiment("credit_scoring_dataframe_evaluation")
    os.makedirs(cache_dir, exist_ok=True)

    for name, frac in balances.items():
        if frac is None:
            df_bal = df.copy()
        else:
            df_bal = resample_for_balance(df, target_col=target_col, positive_fraction=frac, random_state=random_state)

        # Prepare features: drop target, keep numeric features only
        if target_col not in df_bal.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        X = df_bal.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            raise ValueError("No numeric features available after selection for modeling")
        y = df_bal[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])
        pipe.fit(X_train, y_train)
        y_pred_proba_train = pipe.predict_proba(X_train)[:, 1]
        y_pred_proba_test = pipe.predict_proba(X_test)[:, 1]
        train_auc = float(roc_auc_score(y_train, y_pred_proba_train))
        test_auc = float(roc_auc_score(y_test, y_pred_proba_test))
        results[name] = {"train_auc": train_auc, "test_auc": test_auc, "n_rows": len(df_bal), "n_cols": X.shape[1]}

        # MLflow logging for this balance
        with mlflow.start_run(run_name=f"df_eval_{Path(path_parquet).stem}_{name}", nested=False):
            mlflow.log_param("evaluated_balance", name)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("n_rows", len(df_bal))
            mlflow.log_metric("n_cols", X.shape[1])
            # save a small sample artifact
            sample_fp = os.path.join(cache_dir, f"sample_{Path(path_parquet).stem}_{name}.csv")
            df_bal.head(200).to_csv(sample_fp, index=False)
            mlflow.log_artifact(sample_fp, artifact_path="df_samples")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet (from process_df_global)")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    res = evaluate_dataframe(args.input, target_col=args.target, cache_dir=args.cache_dir)
    print(json.dumps(res, indent=2))