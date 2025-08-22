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

def resample_for_balance(df: pd.DataFrame, target_col: str, positive_fraction: float, random_state: int = 42):
    df = df.copy()
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    
    if pos.empty or neg.empty:
        print(f"[resample_for_balance] Warning: pos or neg class empty for target_col='{target_col}'. Returning original df.")
        return df
        
    n_total = len(df)
    n_pos = int(round(positive_fraction * n_total))
    n_pos = max(1, n_pos)
    n_neg = n_total - n_pos
    
    pos_sample = pos.sample(n=n_pos, replace=(n_pos > len(pos)), random_state=random_state)
    neg_sample = neg.sample(n=n_neg, replace=(n_neg > len(neg)), random_state=random_state)
    
    return pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=random_state).reset_index(drop=True)

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
                       persist_parquet: bool = False):
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
    balances = {
        "initial": None,
        "25_75": 0.25,
        "50_50": 0.5
    }
    results = {}
    # ensure all artifacts / mlruns are placed under the external cache
    os.makedirs(cache_dir, exist_ok=True)
    ensure_mlflow_from_env(cache_dir)
    mlflow.set_experiment("credit_scoring_dataframe_evaluation")

    balanced_results = {}
    for name, frac in balances.items():
        print(f"[evaluate_dataframe] Processing balance '{name}' (frac={frac})")
        if frac is None:
            df_bal = df.copy()
        else:
            df_bal = resample_for_balance(df, target_col=target_col, positive_fraction=frac, random_state=random_state)

        out_path = None
        if persist_parquet:
            out_path = os.path.join(cache_dir, f"balanced_{Path(getattr(df_or_path,'name', 'df')).stem}_{name}.parquet")
            df_bal.to_parquet(out_path, engine="pyarrow")
            print(f"[evaluate_dataframe] Saved balanced file to {out_path} (rows={len(df_bal)})")
        else:
            # write only a small sample for MLflow and diagnostics (temp file)
            sample_fp = os.path.join(cache_dir, f"balanced_sample_{Path(getattr(df_or_path,'name', 'df')).stem}_{name}.csv")
            df_bal.head(200).to_csv(sample_fp, index=False)
            mlflow.log_artifact(sample_fp, artifact_path=f"balanced_samples/{name}")

        if target_col not in df_bal.columns:
            msg = f"Target column '{target_col}' not found in dataframe for balance '{name}'"
            print(f"[evaluate_dataframe] ERROR: {msg}")
            results[name] = {"error": msg, "n_rows": len(df_bal)}
            continue

        # Validate that the target is numeric and contains no NaN/inf; process_df_global should have cleaned target.
        y_series = df_bal[target_col]
        y_num = pd.to_numeric(y_series, errors="coerce")
        n_nan = int(y_num.isna().sum())
        n_inf = int(np.isinf(y_num).sum()) if y_num.dtype.kind in "f" else 0
        if n_nan > 0 or n_inf > 0:
            err_msg = (
                f"Target column '{target_col}' contains {n_nan} NaNs and {n_inf} infs in balance '{name}'. "
                "This should be resolved by process_df_global (set cfg.impute to 'drop','median','mode','zero'). "
                f"Diagnostics saved in cache. Parquet: {path_parquet}"
            )
            print(f"[evaluate_dataframe] ERROR: {err_msg}")
            os.makedirs(cache_dir, exist_ok=True)
            diag2 = {"balance": name, "n_nan": n_nan, "n_inf": n_inf}
            diag2_path = os.path.join(cache_dir, f"diag_eval_bad_target_{Path(path_parquet).stem}_{name}.json")
            with open(diag2_path, "w", encoding="utf8") as f:
                json.dump(diag2, f, indent=2)
            results[name] = {"error": "target_not_clean", "n_nan": n_nan, "n_inf": n_inf, "diagnostics": diag2_path}
            continue

        # final target ready for modeling (safe cast)
        y = y_num.astype(int)

        # Prepare X and sanitize numeric features: replace inf with NaN, cap extreme values, and impute medians if needed.
        X = df_bal.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            msg = "No numeric features available after selection for modeling"
            print(f"[evaluate_dataframe] ERROR: {msg}")
            results[name] = {"error": msg, "n_rows": len(df_bal)}
            continue

        # sanitize numeric matrix to avoid "contains infinity or too large" errors
        print(f"[evaluate_dataframe] Sanitizing numeric features (shape={X.shape})")
        # Replace inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        # detect columns with extremely large abs values
        absmax = X.abs().max(skipna=True)
        huge_cols = absmax[absmax > 1e12].index.tolist()
        if huge_cols:
            print(f"[evaluate_dataframe] Warning: {len(huge_cols)} columns contain extremely large values (>1e12). Will clip and save diagnostics: {huge_cols[:10]}")
            # clip them to a reasonable threshold
            X[huge_cols] = X[huge_cols].clip(-1e12, 1e12)

        # After replacement/clipping, fill NaNs with median (robust fallback if process_df_global missed some)
        n_missing_after = int(X.isna().sum().sum())
        if n_missing_after > 0:
            print(f"[evaluate_dataframe] Found {n_missing_after} missing numeric entries after sanitization; filling with column median.")
            medians = X.median(numeric_only=True)
            X = X.fillna(medians)

        # final check for non-finite entries
        if not np.isfinite(X.to_numpy()).all():
            # save diagnostic and fail (should not happen after sanitization)
            diag_fp = os.path.join(cache_dir, f"diag_X_nonfinite_{Path(path_parquet).stem}_{name}.json")
            with open(diag_fp, "w", encoding="utf8") as f:
                json.dump({
                    "balance": name,
                    "absmax_sample": absmax.head(20).to_dict()
                }, f, indent=2)
            raise ValueError(f"After sanitization X still contains non-finite values. Diagnostics saved to {diag_fp}")

        # safe to continue with train/test split and modeling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])
        print(f"[evaluate_dataframe] Fitting model for balance '{name}' (train rows={len(X_train)})")
        pipe.fit(X_train, y_train)
        p_test = pipe.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, p_test))

        # save some artifacts for debugging
        debug_fp = os.path.join(cache_dir, f"df_eval_debug_{safe_stem}_{name}.json")
        debug_info = {
            "path_parquet": path_parquet,
            "balance": name,
            "n_rows_balanced": len(df_bal),
            "n_features": int(X.shape[1]),
            "test_auc": test_auc
        }
        with open(debug_fp, "w", encoding="utf8") as f:
            json.dump(debug_info, f, indent=2)
        print(f"[evaluate_dataframe] Debug info saved -> {debug_fp}")

        # log to mlflow
        with mlflow.start_run(run_name=f"df_eval_{Path(path_parquet).stem}_{name}", nested=False):
            mlflow.log_param("balance", name)
            mlflow.log_param("balance_fraction", str(frac))
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_param("target_col", target_col)
            mlflow.log_metric("n_rows", int(len(df_bal)))
            mlflow.log_metric("n_features", int(X.shape[1]))
            if out_path:
                mlflow.log_artifact(out_path, artifact_path="balanced_dataframes")
            mlflow.log_artifact(debug_fp, artifact_path="diagnostics")

        # at end of loop collect balanced df object
        balanced_results[name] = {
            "df": df_bal,
            "balanced_parquet": out_path,
            "test_auc": test_auc,
            "n_rows": len(df_bal),
            "n_features": X.shape[1],
            "diagnostics": debug_fp
        }

    results = {k: {kk: v for kk, v in vals.items() if kk != "df"} for k, vals in balanced_results.items()}
    # also return the in-memory DataFrames via a separate key for callers that want them
    return {"metrics": results, "balanced_dfs": {k: vals["df"] for k, vals in balanced_results.items()}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet (from process_df_global)")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    res = evaluate_dataframe(args.input, target_col=args.target, cache_dir=args.cache_dir)
    print(json.dumps(res, indent=2))