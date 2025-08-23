# This script has to be modified following these instructions :
# Parameter : Specify Target balance
# Actions :
# - undersample the majority class to achieve the specified balance
# - oversample the minority class to achieve the specified balance (smote)
# - under_over sample half way both classes to achieve the specified balance
# - train a logistic regression model on each balanced dataset
# - evaluate the model using ROC AUC
# - log the results to MLflow

from pathlib import Path
import os
import json
import argparse
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tempfile
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def validate_evaluate_inputs(path_parquet: str, target_col: str, cache_dir: str):
    errs = []
    if path_parquet and not os.path.exists(path_parquet):
        errs.append(f"input parquet not found: {path_parquet}")
        
    # ensure cache_dir writeable (try create)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tf = tempfile.NamedTemporaryFile(dir=cache_dir, delete=True)
        tf.close()
    except Exception as e:
        errs.append(f"cache_dir not writable: {cache_dir} -> {e}")
        
    if errs:
        msg = "evaluate_dataframe input validation failed:\n  " + "\n  ".join(errs)
        print(f"[evaluate_dataframe][VALIDATION] {msg}")
        raise ValueError(msg)
        
    print(f"[evaluate_dataframe][VALIDATION] OK: parquet={path_parquet}, target_col={target_col}, cache_dir={cache_dir}")

def _load_df(df_or_path):
    """Accept either a pandas.DataFrame or a parquet path."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    return pd.read_parquet(df_or_path, engine="pyarrow")

def ensure_mlflow_from_env(cache_dir: str):
    """
    Use MLFLOW_TRACKING_URI if present (set by run_pipeline.configure_mlflow),
    otherwise set a file:// URI under cache_dir (Path.as_uri() for Windows).
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        try:
            mlflow.set_tracking_uri(env_uri)
        except Exception:
            pass
        return mlflow.get_tracking_uri()
        
    # fallback to file under cache_dir
    try:
        mlruns_dir = os.path.join(os.path.abspath(cache_dir), "mlruns")
        mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
    except Exception:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(cache_dir)}")
    return mlflow.get_tracking_uri()

def evaluate_dataframe(df_or_path, target_col: str = "TARGET", test_size: float = 0.2,
                       random_state: int = 42, cache_dir: str = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\cache",
                       persist_parquet: bool = False, target_balance: float = 0.5):
    """
    df_or_path: either a pandas.DataFrame or a path to parquet.
    """
    # validate inputs (will accept path or None if df provided)
    validate_evaluate_inputs(df_or_path if not isinstance(df_or_path, pd.DataFrame) else None, target_col, cache_dir)

    # ensure module uses the centralized mlflow tracking (set by run_pipeline)
    try:
        ensure_mlflow_from_env(cache_dir)
    except Exception:
        pass

    # normalize an identifier for filenames/logging: prefer the path when provided, else dataframe name or 'df'
    if isinstance(df_or_path, pd.DataFrame):
        path_parquet = getattr(df_or_path, "name", "df")
        df = df_or_path.copy()
    else:
        path_parquet = df_or_path
        df = pd.read_parquet(df_or_path, engine="pyarrow")

    # safe stem for debug filenames
    try:
        safe_stem = Path(path_parquet).stem if isinstance(path_parquet, str) else str(path_parquet)
    except Exception:
        safe_stem = "df"

    print(f"[evaluate_dataframe] Loading parquet: {path_parquet}")
    print(f"[evaluate_dataframe] Loaded: rows={len(df)}, cols={len(df.columns)}")
    
    # Define resampling strategies
    strategies = {
        "initial": None,
        "undersample": RandomUnderSampler(sampling_strategy=target_balance, random_state=random_state),
        "oversample_smote": SMOTE(sampling_strategy=target_balance, random_state=random_state),
        "under_over_sample": ImbPipeline(steps=[
            ('under', RandomUnderSampler(sampling_strategy=0.2, random_state=random_state)),
            ('over', SMOTE(sampling_strategy=target_balance, random_state=random_state))
        ])
    }

    results = {}
    # ensure all artifacts / mlruns are placed under the external cache
    os.makedirs(cache_dir, exist_ok=True)
    ensure_mlflow_from_env(cache_dir)
    mlflow.set_experiment("credit_scoring_dataframe_evaluation")

    # Split data into training and testing sets to prevent data leakage
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    balanced_results = {}
    for name, strategy in strategies.items():
        print(f"[evaluate_dataframe] Processing strategy '{name}'")

        X_resampled, y_resampled = X_train, y_train
        if strategy:
            try:
                X_resampled, y_resampled = strategy.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Error during resampling for strategy {name}: {e}")
                continue
        
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

        out_path = None
        if persist_parquet:
            out_path = os.path.join(cache_dir, f"balanced_{safe_stem}_{name}.parquet")
            df_resampled.to_parquet(out_path, engine="pyarrow")
            print(f"[evaluate_dataframe] Saved balanced file to {out_path} (rows={len(df_resampled)})")
        else:
            # write only a small sample for MLflow and diagnostics (temp file)
            sample_fp = os.path.join(cache_dir, f"balanced_sample_{safe_stem}_{name}.csv")
            df_resampled.head(200).to_csv(sample_fp, index=False)
            mlflow.log_artifact(sample_fp, artifact_path=f"balanced_samples/{name}")

        # Prepare X and sanitize numeric features: replace inf with NaN, cap extreme values, and impute medians if needed.
        X_resampled = X_resampled.select_dtypes(include=[np.number])
        if X_resampled.shape[1] == 0:
            msg = "No numeric features available after selection for modeling"
            print(f"[evaluate_dataframe] ERROR: {msg}")
            results[name] = {"error": msg, "n_rows": len(df_resampled)}
            continue

        # sanitize numeric matrix to avoid "contains infinity or too large" errors
        print(f"[evaluate_dataframe] Sanitizing numeric features (shape={X_resampled.shape})")
        X_resampled = X_resampled.replace([np.inf, -np.inf], np.nan)
        absmax = X_resampled.abs().max(skipna=True)
        huge_cols = absmax[absmax > 1e12].index.tolist()
        if huge_cols:
            print(f"[evaluate_dataframe] Warning: {len(huge_cols)} columns contain extremely large values (>1e12). Will clip and save diagnostics: {huge_cols[:10]}")
            X_resampled[huge_cols] = X_resampled[huge_cols].clip(-1e12, 1e12)

        n_missing_after = int(X_resampled.isna().sum().sum())
        if n_missing_after > 0:
            print(f"[evaluate_dataframe] Found {n_missing_after} missing numeric entries after sanitization; filling with column median.")
            medians = X_resampled.median(numeric_only=True)
            X_resampled = X_resampled.fillna(medians)

        if not np.isfinite(X_resampled.to_numpy()).all():
            diag_fp = os.path.join(cache_dir, f"diag_X_nonfinite_{safe_stem}_{name}.json")
            with open(diag_fp, "w", encoding="utf8") as f:
                json.dump({"balance": name, "absmax_sample": absmax.head(20).to_dict()}, f, indent=2)
            raise ValueError(f"After sanitization X still contains non-finite values. Diagnostics saved to {diag_fp}")

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])
        print(f"[evaluate_dataframe] Fitting model for strategy '{name}' (train rows={len(X_resampled)})")
        pipe.fit(X_resampled, y_resampled)
        
        # Evaluate on the original test set
        X_test_numeric = X_test.select_dtypes(include=[np.number])
        p_test = pipe.predict_proba(X_test_numeric)[:, 1]
        test_auc = float(roc_auc_score(y_test, p_test))

        debug_fp = os.path.join(cache_dir, f"df_eval_debug_{safe_stem}_{name}.json")
        debug_info = {
            "path_parquet": path_parquet,
            "strategy": name,
            "n_rows_resampled": len(df_resampled),
            "n_features": int(X_resampled.shape[1]),
            "test_auc": test_auc
        }
        with open(debug_fp, "w", encoding="utf8") as f:
            json.dump(debug_info, f, indent=2)
        print(f"[evaluate_dataframe] Debug info saved -> {debug_fp}")

        with mlflow.start_run(run_name=f"df_eval_{safe_stem}_{name}", nested=True):
            mlflow.log_param("strategy", name)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_param("target_col", target_col)
            mlflow.log_metric("n_rows_resampled", int(len(df_resampled)))
            mlflow.log_metric("n_features", int(X_resampled.shape[1]))
            if out_path:
                mlflow.log_artifact(out_path, artifact_path="balanced_dataframes")
            mlflow.log_artifact(debug_fp, artifact_path="diagnostics")

        balanced_results[name] = {
            "df": df_resampled,
            "balanced_parquet": out_path,
            "test_auc": test_auc,
            "n_rows": len(df_resampled),
            "n_features": X_resampled.shape[1],
            "diagnostics": debug_fp
        }

    results = {k: {kk: v for kk, v in vals.items() if kk != "df"} for k, vals in balanced_results.items()}
    return {"metrics": results, "balanced_dfs": {k: vals["df"] for k, vals in balanced_results.items()}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet (from process_df_global)")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--target-balance", type=float, default=0.5, help="Target balance for the positive class")
    args = parser.parse_args()
    res = evaluate_dataframe(args.input, target_col=args.target, cache_dir=args.cache_dir, target_balance=args.target_balance)
    print(json.dumps(res["metrics"], indent=2))