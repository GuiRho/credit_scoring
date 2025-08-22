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

def resample_for_balance(df: pd.DataFrame, target_col: str, positive_fractions: list, random_state: int = 42):
    """
    Return a dictionary of DataFrames, each balanced to a different positive_fraction.
    Uses sampling with replacement if needed to reach the requested fraction while preserving n_total.
    """
    df = df.copy()
    rng = np.random.RandomState(random_state)
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    n_total = len(df)
    print(f"[DEBUG] resample_for_balance: Original positive samples: {len(pos)}, negative samples: {len(neg)}, total: {n_total}")
    balanced_dfs = {}

    for frac in positive_fractions:
        n_pos_desired = max(1, int(round(frac * n_total)))
        n_neg_desired = max(0, n_total - n_pos_desired)

        print(f"[DEBUG] resample_for_balance: For fraction {frac*100}%: desired positive={n_pos_desired}, desired negative={n_neg_desired}")

        if len(pos) == 0 and n_pos_desired > 0:
            raise ValueError(f"No positive samples available for fraction {frac}")
        if len(neg) == 0 and n_neg_desired > 0:
            raise ValueError(f"No negative samples available for fraction {frac}")

        if len(pos) >= n_pos_desired:
            pos_sample = pos.sample(n=n_pos_desired, replace=False, random_state=rng)
        else:
            pos_sample = pos.sample(n=n_pos_desired, replace=True, random_state=rng)
        print(f"[DEBUG] resample_for_balance: Positive sample size: {len(pos_sample)}")

        if len(neg) >= n_neg_desired:
            neg_sample = neg.sample(n=n_neg_desired, replace=False, random_state=rng)
        else:
            neg_sample = neg.sample(n=n_neg_desired, replace=True, random_state=rng)
        print(f"[DEBUG] resample_for_balance: Negative sample size: {len(neg_sample)}")

        df_bal = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1.0, random_state=rng).reset_index(drop=True)
        print(f"[DEBUG] resample_for_balance: Balanced dataframe shape for frac_{int(frac*100)}: {df_bal.shape}")
        balanced_dfs[f"frac_{int(frac*100)}"] = df_bal

    return balanced_dfs

def validate_evaluate_inputs(path_parquet: str, target_col: str, cache_dir: str):
    errs = []
    if path_parquet and not os.path.exists(path_parquet):
        errs.append(f"Input parquet not found: {path_parquet}")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tf = tempfile.NamedTemporaryFile(dir=cache_dir, delete=True)
        tf.close()
    except Exception as e:
        errs.append(f"Cache_dir not writable: {cache_dir} -> {e}")

    if errs:
        msg = "Input validation failed:\n  " + "\n  ".join(errs)
        print(f"[ERROR] {msg}")
        raise ValueError(msg)

    print(f"[INFO] Input validation OK: parquet={path_parquet}, target_col={target_col}, cache_dir={cache_dir}")

def _load_df(df_or_path):
    """Accept either a pandas.DataFrame or a parquet path."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    return pd.read_parquet(df_or_path, engine="pyarrow")

def ensure_mlflow_from_env(cache_dir: str):
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        try:
            mlflow.set_tracking_uri(env_uri)
        except Exception:
            pass
        return mlflow.get_tracking_uri()

    try:
        mlruns_dir = os.path.join(os.path.abspath(cache_dir), "mlruns")
        mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
    except Exception:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(cache_dir)}")
    return mlflow.get_tracking_uri()

def ensure_no_active_mlflow_run():
    try:
        ar = mlflow.active_run()
        if ar is not None:
            print(f"[INFO] Ending previously active mlflow run id={ar.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        print(f"[WARNING] Could not check/end active mlflow run: {e}")

def evaluate_dataframe(df_or_path, target_col: str = "TARGET", test_size: float = 0.2,
                       random_state: int = 42, cache_dir: str = r"cache",
                       persist_parquet: bool = False):
    validate_evaluate_inputs(df_or_path if not isinstance(df_or_path, pd.DataFrame) else None, target_col, cache_dir)

    try:
        uri = ensure_mlflow_from_env(cache_dir)
        print(f"[INFO] MLflow tracking URI: {uri}")
    except Exception:
        print("[WARNING] Could not ensure mlflow tracking from env")

    ensure_no_active_mlflow_run()

    if isinstance(df_or_path, pd.DataFrame):
        path_parquet = getattr(df_or_path, "name", "df")
        df = df_or_path.copy()
    else:
        path_parquet = df_or_path
        df = pd.read_parquet(df_or_path, engine="pyarrow")

    try:
        safe_stem = Path(path_parquet).stem if isinstance(path_parquet, str) else str(path_parquet)
    except Exception:
        safe_stem = "df"

    print(f"[INFO] Loading parquet: {path_parquet}")
    print(f"[INFO] Loaded: rows={len(df)}, cols={len(df.columns)}")

    positive_fractions = [frac / 100 for frac in range(10, 51, 5)]
    balanced_dfs = resample_for_balance(df, target_col=target_col, positive_fractions=positive_fractions, random_state=random_state)

    results = {}
    os.makedirs(cache_dir, exist_ok=True)
    ensure_mlflow_from_env(cache_dir)
    mlflow.set_experiment("credit_scoring_dataframe_evaluation")
    exp = mlflow.get_experiment_by_name("credit_scoring_dataframe_evaluation")
    print(f"[INFO] MLflow experiment id: {exp.experiment_id if exp else 'N/A'}")

    for name, df_bal in balanced_dfs.items():
        frac = int(name.replace("frac_", "")) / 100
        print(f"\n[INFO] Processing balance '{name}' (frac={frac:.2f})")
        print(f"[INFO] Balance '{name}' dataframe shape: rows={len(df_bal)}, cols={len(df_bal.columns)}")

        out_path = None
        if persist_parquet:
            out_path = os.path.join(cache_dir, f"balanced_{safe_stem}_{name}.parquet")
            df_bal.to_parquet(out_path, engine="pyarrow")
            print(f"[INFO] Saved balanced file to {out_path} (rows={len(df_bal)})")
        else:
            sample_fp = os.path.join(cache_dir, f"balanced_sample_{safe_stem}_{name}.csv")
            df_bal.head(200).to_csv(sample_fp, index=False)
            try:
                mlflow.log_artifact(sample_fp, artifact_path=f"balanced_samples/{name}")
                print(f"[INFO] Logged balanced sample to mlflow: {sample_fp}")
            except Exception as e:
                print(f"[WARNING] Could not log balanced sample to mlflow: {e}")

        if target_col not in df_bal.columns:
            msg = f"Target column '{target_col}' not found in dataframe for balance '{name}'"
            print(f"[ERROR] {msg}")
            results[name] = {"error": msg, "n_rows": len(df_bal)}
            continue

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
            print(f"[ERROR] {err_msg}")
            diag2 = {"balance": name, "n_nan": n_nan, "n_inf": n_inf}
            diag2_path = os.path.join(cache_dir, f"diag_eval_bad_target_{safe_stem}_{name}.json")
            with open(diag2_path, "w", encoding="utf8") as f:
                json.dump(diag2, f, indent=2)
            print(f"[INFO] Wrote bad-target diagnostic: {diag2_path}")
            results[name] = {"error": "target_not_clean", "n_nan": n_nan, "n_inf": n_inf, "diagnostics": diag2_path}
            continue

        y = y_num.astype(int)
        X = df_bal.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            msg = "No numeric features available after selection for modeling"
            print(f"[ERROR] {msg}")
            results[name] = {"error": msg, "n_rows": len(df_bal)}
            continue

        print(f"[INFO] Sanitizing numeric features (shape={X.shape})")
        X = X.replace([np.inf, -np.inf], np.nan)
        absmax = X.abs().max(skipna=True)
        huge_cols = absmax[absmax > 1e12].index.tolist()
        if huge_cols:
            print(f"[WARNING] {len(huge_cols)} columns contain extremely large values (>1e12). Will clip and save diagnostics: {huge_cols[:10]}")
            abs_diag_fp = os.path.join(cache_dir, f"diag_absmax_{safe_stem}_{name}.json")
            with open(abs_diag_fp, "w", encoding="utf8") as f:
                json.dump({"absmax_sample": absmax.head(50).to_dict()}, f, indent=2)
            print(f"[INFO] Wrote absmax diagnostics -> {abs_diag_fp}")
            X[huge_cols] = X[huge_cols].clip(-1e12, 1e12)

        n_missing_after = int(X.isna().sum().sum())
        if n_missing_after > 0:
            print(f"[INFO] Found {n_missing_after} missing numeric entries after sanitization; filling with column median.")
            medians = X.median(numeric_only=True)
            print(f"[DEBUG] Medians calculated from balanced dataframe for imputation: {medians.head()}")
            medians_fp = os.path.join(cache_dir, f"medians_{safe_stem}_{name}.json")
            try:
                medians.to_json(medians_fp)
                print(f"[INFO] Saved medians -> {medians_fp}")
            except Exception:
                pass
            X = X.fillna(medians)

        if not np.isfinite(X.to_numpy()).all():
            diag_fp = os.path.join(cache_dir, f"diag_X_nonfinite_{safe_stem}_{name}.json")
            with open(diag_fp, "w", encoding="utf8") as f:
                json.dump({
                    "balance": name,
                    "absmax_sample": absmax.head(20).to_dict()
                }, f, indent=2)
            print(f"[ERROR] Non-finite values remain. Diagnostics saved to {diag_fp}")
            raise ValueError(f"After sanitization X still contains non-finite values. Diagnostics saved to {diag_fp}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])
        print(f"[INFO] Fitting model for balance '{name}' (train rows={len(X_train)})")
        pipe.fit(X_train, y_train)
        p_test = pipe.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, p_test))

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
        print(f"[INFO] Debug info saved -> {debug_fp}")

        ensure_no_active_mlflow_run()
        with mlflow.start_run(run_name=f"df_eval_{safe_stem}_{name}", nested=False):
            mlflow.log_param("balance", name)
            mlflow.log_param("balance_fraction", str(frac))
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_param("target_col", target_col)
            mlflow.log_metric("n_rows", int(len(df_bal)))
            mlflow.log_metric("n_features", int(X.shape[1]))
            if out_path:
                try:
                    mlflow.log_artifact(out_path, artifact_path="balanced_dataframes")
                except Exception as e:
                    print(f"[WARNING] Could not log balanced parquet to mlflow: {e}")
            try:
                mlflow.log_artifact(debug_fp, artifact_path="diagnostics")
            except Exception as e:
                print(f"[WARNING] Could not log debug artifact: {e}")

        results[name] = {
            "test_auc": test_auc,
            "n_rows": len(df_bal),
            "n_features": X.shape[1],
            "diagnostics": debug_fp
        }

    # Print all results
    print("\n=== ALL RESULTS ===")
    for name, metrics in results.items():
        if "test_auc" in metrics:
            print(f"{name}: AUC={metrics['test_auc']:.4f}, Features={metrics['n_features']}, Rows={metrics['n_rows']}")

    # Identify and print the best model
    valid_results = {k: v for k, v in results.items() if "test_auc" in v}
    if valid_results:
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k]["test_auc"])
        best_metrics = valid_results[best_name]

        print("\n=== BEST MODEL ===")
        print(f"Best target distribution: {best_name.replace('frac_', '')}%")
        print(f"Test AUC: {best_metrics['test_auc']:.4f}")
        print(f"Number of features: {best_metrics['n_features']}")
        print(f"Number of rows: {best_metrics['n_rows']}")
        print("==================\n")
    else:
        print("\n[ERROR] No valid results to compare.")

    # NEW: Save the full results dictionary to a JSON file
    all_metrics_fp = os.path.join(cache_dir, "all_dataframe_metrics.json")
    with open(all_metrics_fp, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] All dataframe evaluation metrics saved to: {all_metrics_fp}")

    return {"metrics": results, "balanced_dfs": balanced_dfs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet (from process_df_global)")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--persist-parquet", action="store_true", help="Persist balanced dataframes as parquet files")
    args = parser.parse_args()
    print(f"[CLI] args: input={args.input}, target={args.target}, cache_dir={args.cache_dir}")
    res = evaluate_dataframe(args.input, target_col=args.target, cache_dir=args.cache_dir, persist_parquet=args.persist_parquet)

    if isinstance(res, dict) and "metrics" in res:
        print(json.dumps(res["metrics"], indent=2))
        if "balanced_dfs" in res:
            summary = {k: {"rows": int(v.shape[0]), "cols": int(v.shape[1])} for k, v in res["balanced_dfs"].items()}
            print("[CLI] balanced_dfs summary:", json.dumps(summary, indent=2))
    else:
        try:
            print(json.dumps(res, indent=2, default=str))
        except Exception:
            print(str(res))
