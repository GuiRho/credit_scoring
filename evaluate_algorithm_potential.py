from pathlib import Path
import os
import json
import argparse
import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

CLASSIFIERS = {
    "logreg": LogisticRegression(max_iter=1000, solver="liblinear"),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(),
}
if XGB_AVAILABLE:
    CLASSIFIERS["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

def _best_threshold_max_recall(y_true, y_pred_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_rec = -1.0
    for t in thresholds:
        r = recall_score(y_true, (y_pred_proba >= t).astype(int))
        if r > best_rec:
            best_rec = r
            best_t = t
    return best_t, best_rec

def _compute_custom_and_normalized(y_true, y_pred_bin, pos_proportion):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    custom = 2 * tp + 1 * tn - 1 * fp - 10 * fn
    n = len(y_true)
    pos_prop = max(pos_proportion, 1e-9)
    normalized = custom * (1.0 / max(1, n)) * (1.0 / pos_prop)
    return float(custom), float(normalized)

def validate_algorithm_inputs(path_parquet: str, target_col: str, cache_dir: str):
    errs = []
    if path_parquet and not os.path.exists(path_parquet):
        errs.append(f"input parquet not found: {path_parquet}")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        errs.append(f"cannot prepare cache_dir: {cache_dir} -> {e}")
    if errs:
        msg = "evaluate_algorithms input validation failed:\n  " + "\n  ".join(errs)
        print(f"[evaluate_algorithms][VALIDATION] {msg}")
        raise ValueError(msg)
    print(f"[evaluate_algorithms][VALIDATION] OK: parquet={path_parquet}, target={target_col}, cache={cache_dir}")

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

def _load_df(df_or_path):
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    return pd.read_parquet(df_or_path, engine="pyarrow")

def load_best_balanced_df(metrics_file: str, balanced_dfs_dir: str):
    """
    Load the best balanced DataFrame from the previous script's output.
    metrics_file: Path to the JSON file containing metrics from the previous script.
    balanced_dfs_dir: Directory containing the balanced DataFrames.
    """
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    valid_results = {k: v for k, v in metrics.items() if "test_auc" in v}
    if not valid_results:
        raise ValueError("No valid results found in metrics file.")

    best_name = max(valid_results.keys(), key=lambda k: valid_results[k]["test_auc"])
    best_metrics = valid_results[best_name]
    best_frac = int(best_name.replace("frac_", ""))

    print(f"[INFO] Best target distribution: {best_frac}% (AUC: {best_metrics['test_auc']:.4f})")

    balanced_df_path = os.path.join(balanced_dfs_dir, f"balanced_{Path(metrics_file).stem}_{best_name}.parquet")
    if not os.path.exists(balanced_df_path):
        raise FileNotFoundError(f"Balanced DataFrame not found at: {balanced_df_path}")

    df = pd.read_parquet(balanced_df_path, engine="pyarrow")
    print(f"[INFO] Loaded best balanced DataFrame: {balanced_df_path} (rows={len(df)}, cols={len(df.columns)})")
    return df, best_frac, best_metrics

def evaluate_algorithms(df_or_path, target_col: str = "TARGET", test_size: float = 0.2,
                        random_state: int = 42, cache_dir: str = r"cache",
                        metrics_file: str = None, balanced_dfs_dir: str = None):
    """
    Accept DataFrame or path. If metrics_file and balanced_dfs_dir are provided,
    load the best balanced DataFrame from the previous script.
    """
    if metrics_file and balanced_dfs_dir:
        df = load_best_balanced_df(metrics_file, balanced_dfs_dir)
        path_parquet = f"best_balanced_{df[1]}"
    elif isinstance(df_or_path, pd.DataFrame):
        path_parquet = "dataframe_in_memory"
        df = df_or_path.copy()
    else:
        path_parquet = df_or_path
        df = _load_df(path_parquet)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric features available after selection for modeling")

    y = df[target_col].astype(int)
    pos_prop_global = float(y.mean()) if len(y) > 0 else 0.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        ensure_mlflow_from_env(cache_dir)
    except Exception:
        pass

    mlflow.set_experiment("credit_scoring_algorithm_evaluation")
    summary = {}
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        p_train = pipe.predict_proba(X_train)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
        best_t, best_rec = _best_threshold_max_recall(y_train, p_train)
        y_train_bin = (p_train >= best_t).astype(int)
        y_test_bin = (p_test >= best_t).astype(int)
        train_custom, train_norm = _compute_custom_and_normalized(y_train, y_train_bin, pos_prop_global)
        test_custom, test_norm = _compute_custom_and_normalized(y_test, y_test_bin, pos_prop_global)
        train_auc = float(roc_auc_score(y_train, p_train))
        test_auc = float(roc_auc_score(y_test, p_test))
        train_acc = float(accuracy_score(y_train, y_train_bin))
        test_acc = float(accuracy_score(y_test, y_test_bin))
        summary[name] = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_threshold_train": float(best_t),
            "train_custom": float(train_custom),
            "test_custom": float(test_custom),
            "train_normalized_custom": float(train_norm),
            "test_normalized_custom": float(test_norm),
            "n_rows": len(df),
            "n_cols": X.shape[1]
        }

        run_name_stem = Path(path_parquet).stem if path_parquet != "dataframe_in_memory" else "df_in_memory"
        with mlflow.start_run(run_name=f"algo_{run_name_stem}_{name}", nested=False):
            mlflow.log_param("algorithm", name)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("test_acc", test_acc)
            mlflow.log_metric("train_custom", train_custom)
            mlflow.log_metric("test_custom", test_custom)
            mlflow.log_metric("train_normalized_custom", train_norm)
            mlflow.log_metric("test_normalized_custom", test_norm)
            mlflow.log_metric("best_threshold_train", float(best_t))
            model_fp = os.path.join(cache_dir, f"model_{run_name_stem}_{name}.joblib")
            os.makedirs(os.path.dirname(model_fp) or ".", exist_ok=True)
            joblib.dump(pipe, model_fp)
            mlflow.log_artifact(model_fp, artifact_path="models")

    # Print summary of all algorithms
    print("\n=== ALGORITHM PERFORMANCE SUMMARY ===")
    for algo, metrics in summary.items():
        print(f"{algo.upper()}:")
        print(f"  Test AUC: {metrics['test_auc']:.4f}")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Test Custom Score: {metrics['test_custom']:.2f}")
        print(f"  Test Normalized Custom Score: {metrics['test_normalized_custom']:.2f}")
        print(f"  Best Threshold: {metrics['best_threshold_train']:.3f}")
        print()

    # Identify and print the best algorithm
    best_algo = max(summary.keys(), key=lambda k: summary[k]["test_normalized_custom"])
    best_metrics = summary[best_algo]
    print("=== BEST ALGORITHM ===")
    print(f"Algorithm: {best_algo.upper()}")
    print(f"Test AUC: {best_metrics['test_auc']:.4f}")
    print(f"Test Accuracy: {best_metrics['test_acc']:.4f}")
    print(f"Test Custom Score: {best_metrics['test_custom']:.2f}")
    print(f"Test Normalized Custom Score: {best_metrics['test_normalized_custom']:.2f}")
    print(f"Best Threshold: {best_metrics['best_threshold_train']:.3f}")
    print("=======================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to processed parquet (from process_df_global)")
    parser.add_argument("--metrics-file", help="Path to the JSON metrics file from the previous script")
    parser.add_argument("--balanced-dfs-dir", help="Directory containing the balanced DataFrames")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.metrics_file and args.balanced_dfs_dir:
        res = evaluate_algorithms(
            None, target_col=args.target, test_size=args.test_size,
            random_state=args.random_state, cache_dir=args.cache_dir,
            metrics_file=args.metrics_file, balanced_dfs_dir=args.balanced_dfs_dir
        )
    else:
        res = evaluate_algorithms(
            args.input, target_col=args.target, test_size=args.test_size,
            random_state=args.random_state, cache_dir=args.cache_dir
        )

    print(json.dumps(res, indent=2))

