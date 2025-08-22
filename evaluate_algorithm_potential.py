import os
import json
import mlflow
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

CLASSIFIERS = {
    "logreg": LogisticRegression(max_iter=1000, solver="liblinear"),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(),
}
if XGB_AVAILABLE:
    CLASSIFIERS["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

def evaluate_algorithms(path_parquet: str, target_col: str = "TARGET", test_size: float = 0.2, random_state: int = 42, cache_dir: str = "cache"):
    df = pd.read_parquet(path_parquet, engine='pyarrow')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("No numeric features available after selection for modeling")
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    mlflow.set_experiment("credit_scoring_algorithm_evaluation")
    os.makedirs(cache_dir, exist_ok=True)
    summary = {}
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_train_p = pipe.predict_proba(X_train)[:, 1]
        y_test_p = pipe.predict_proba(X_test)[:, 1]
        train_auc = float(roc_auc_score(y_train, y_train_p))
        test_auc = float(roc_auc_score(y_test, y_test_p))
        train_acc = float(accuracy_score(y_train, (y_train_p >= 0.5).astype(int)))
        test_acc = float(accuracy_score(y_test, (y_test_p >= 0.5).astype(int)))
        summary[name] = {"train_auc": train_auc, "test_auc": test_auc, "train_acc": train_acc, "test_acc": test_acc}

        with mlflow.start_run(run_name=f"algo_{Path(path_parquet).stem}_{name}", nested=False):
            mlflow.log_param("algorithm", name)
            mlflow.log_param("input_parquet", path_parquet)
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("test_acc", test_acc)
            # save model artifact
            model_fp = os.path.join(cache_dir, f"model_{Path(path_parquet).stem}_{name}.joblib")
            os.makedirs(os.path.dirname(model_fp) or ".", exist_ok=True)
            import joblib
            joblib.dump(pipe, model_fp)
            mlflow.log_artifact(model_fp, artifact_path="models")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    s = evaluate_algorithms(args.input, target_col=args.target, cache_dir=args.cache_dir)
    print(json.dumps(s, indent=2))