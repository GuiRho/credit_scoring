import os
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def get_classifiers(random_state: int = 42) -> Dict[str, object]:
    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=8),
    }
    if XGB_AVAILABLE:
        classifiers["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, n_jobs=-1)
    if CATBOOST_AVAILABLE:
        classifiers["catboost"] = CatBoostClassifier(n_estimators=100, max_depth=8, random_state=random_state, verbose=100)
    return classifiers


def _compute_custom_and_normalized(y_true: np.ndarray, y_pred_bin: np.ndarray, pos_proportion: float) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    custom = (2 * tp) + (1 * tn) - (1 * fp) - (10 * fn)
    n = len(y_true)
    pos_prop = max(pos_proportion, 1e-9)
    normalized = custom * (1.0 / max(1, n)) * (1.0 / pos_prop)
    return float(custom), float(normalized)


def _find_best_threshold_custom_score(y_true: np.ndarray, y_pred_proba: np.ndarray, pos_proportion: float) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -np.inf
    for t in thresholds:
        y_pred_bin = (y_pred_proba >= t).astype(int)
        _, norm_score = _compute_custom_and_normalized(y_true, y_pred_bin, pos_proportion)
        if norm_score > best_score:
            best_score = norm_score
            best_t = t
    return best_t, best_score


def evaluate_algorithms(input_dir: str, target_col: str, random_state: int):
    print(f"\n{'=' * 20} Processing Dataset: {input_dir} {'=' * 20}")
    train_path = os.path.join(input_dir, "train_processed.parquet")
    test_path = os.path.join(input_dir, "test_processed.parquet")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Train/test parquet files not found in directory: {input_dir}")
    df_train = pd.read_parquet(train_path, engine="pyarrow")
    df_test = pd.read_parquet(test_path, engine="pyarrow")
    X_train_orig = df_train.drop(columns=[target_col])
    y_train_orig = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    int_cols = X_train_orig.select_dtypes(include=['int32', 'int64']).columns
    X_train_orig[int_cols] = X_train_orig[int_cols].astype('float64')
    X_test[int_cols] = X_test[int_cols].astype('float64')
    pos_prop_global = float(y_train_orig.mean())
    target_balances = ['init', 0.11, 0.15]
    for balance in target_balances:
        print(f"\n--- Applying Balance Strategy: {balance} ---")
        X_train, y_train = X_train_orig.copy(), y_train_orig.copy()
        if balance != 'init':
            sampling_ratio = balance / (1 - balance)
            rus = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            print(f"Undersampled training data to {len(X_train)} rows.")
        else:
            print("Using initial training data balance.")
        classifiers = get_classifiers(random_state)
        for name, clf in classifiers.items():
            run_name_stem = Path(input_dir).name
            run_name = f"{run_name_stem}__{name}__bal_{balance}"
            print(f"\nStarting run: {run_name}")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"dataset": run_name_stem, "algorithm": name, "target_balance": str(balance), "input_dir": input_dir})
                mlflow.log_params({f"model_{k}": v for k, v in clf.get_params().items()})
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                print(f"Fitting pipeline for {name}...")
                pipe.fit(X_train, y_train)
                print("Predicting probabilities...")
                p_train = pipe.predict_proba(X_train)[:, 1]
                p_test = pipe.predict_proba(X_test)[:, 1]
                print("Optimizing threshold...")
                best_t, _ = _find_best_threshold_custom_score(y_train.to_numpy(), p_train, pos_prop_global)
                y_train_bin = (p_train >= best_t).astype(int)
                y_test_bin = (p_test >= best_t).astype(int)
                print("Calculating metrics...")
                train_custom, train_norm = _compute_custom_and_normalized(y_train.to_numpy(), y_train_bin, pos_prop_global)
                test_custom, test_norm = _compute_custom_and_normalized(y_test.to_numpy(), y_test_bin, pos_prop_global)
                metrics = {
                    "train_auc": float(roc_auc_score(y_train, p_train)),
                    "test_auc": float(roc_auc_score(y_test, p_test)),
                    "train_accuracy": float(accuracy_score(y_train, y_train_bin)),
                    "test_accuracy": float(accuracy_score(y_test, y_test_bin)),
                    "train_recall_at_best_t": float(recall_score(y_train, y_train_bin)),
                    "test_recall_at_best_t": float(recall_score(y_test, y_test_bin)),
                    "train_custom_score": train_custom,
                    "test_custom_score": test_custom,
                    "train_normalized_custom_score": train_norm,
                    "test_normalized_custom_score": test_norm,
                    "best_threshold": float(best_t),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "num_features": X_train.shape[1]
                }
                mlflow.log_metrics(metrics)
                input_example = X_train.head(5)
                signature = infer_signature(input_example, pipe.predict_proba(input_example))
                mlflow.sklearn.log_model(
                    sk_model=pipe, artifact_path="model", signature=signature,
                    input_example=input_example, metadata={"best_threshold": float(best_t)}
                )
                print(f"  --> Run complete. Test AUC: {metrics['test_auc']:.4f}, Test Normalized Score: {metrics['test_normalized_custom_score']:.4f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate multiple algorithms on multiple datasets and balancing strategies.")
    parser.add_argument("--input-dirs", required=True, nargs='+', help="Paths to directories containing train_processed.parquet and test_processed.parquet.")
    parser.add_argument("--target", default="TARGET", help="Name of the target column.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment", default="Algorithm Comparison", help="MLflow experiment name.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    args = parser.parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    print(f"MLflow is configured to track experiments to: {mlflow.get_tracking_uri()}")
    for input_directory in args.input_dirs:
        try:
            evaluate_algorithms(input_dir=input_directory, target_col=args.target, random_state=args.random_state)
        except Exception as e:
            print(f"Failed to process directory {input_directory}. Error: {e}")
    print(f"\n{'=' * 20} Evaluation Complete {'=' * 20}")
