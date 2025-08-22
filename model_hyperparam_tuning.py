import os
from pathlib import Path
import joblib
import json
import mlflow
import optuna
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# optional backends
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb  # type: ignore
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

def ensure_dir(fp):
    d = os.path.dirname(fp) or "."
    os.makedirs(d, exist_ok=True)
    return fp

def make_pipeline_for(name, params, random_state):
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, solver="liblinear", **(params or {}))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(**(params or {}), random_state=random_state)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "xgboost" and XGB_AVAILABLE:
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, **(params or {}))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "lightgbm" and LGB_AVAILABLE:
        clf = lgb.LGBMClassifier(random_state=random_state, **(params or {}))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unsupported model or missing dependency: {name}")

def tune_model(path_parquet: str, model_name: str, target_col: str = "TARGET", n_trials: int = 50, random_state: int = 42, cache_dir: str = "cache"):
    """
    Generic tuning entry. Supports: random_forest, logreg, xgboost (if installed), lightgbm (if installed).
    Returns (optuna.Study, fitted_pipeline).
    """
    df = pd.read_parquet(path_parquet, engine="pyarrow")
    X = df.drop(columns=[target_col])
    # prefer numeric features only for modeling (safe guard)
    X = X.select_dtypes(include=[np.number])
    y = df[target_col].astype(int)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        pipe = make_pipeline_for("random_forest", params, random_state)
        return float(np.mean(cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    def objective_logreg(trial):
        C = trial.suggest_loguniform("C", 1e-4, 1e2)
        params = {"C": C, "penalty": "l2"}
        pipe = make_pipeline_for("logreg", params, random_state)
        return float(np.mean(cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        pipe = make_pipeline_for("xgboost", params, random_state)
        return float(np.mean(cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        pipe = make_pipeline_for("lightgbm", params, random_state)
        return float(np.mean(cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    objective_map = {
        "random_forest": objective_rf,
        "logreg": objective_logreg,
    }
    if XGB_AVAILABLE:
        objective_map["xgboost"] = objective_xgb
    if LGB_AVAILABLE:
        objective_map["lightgbm"] = objective_lgb

    if model_name not in objective_map:
        raise ValueError(f"Model '{model_name}' cannot be tuned (available: {list(objective_map.keys())})")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective_map[model_name], n_trials=n_trials)

    best_params = study.best_params
    # build final pipeline and fit on full data
    final_pipe = make_pipeline_for(model_name, best_params, random_state)
    final_pipe.fit(X, y)

    # Log to MLflow
    mlflow.set_experiment("credit_scoring_hyperparam_tuning")
    run_name = f"tune_{model_name}_{Path(path_parquet).stem}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_parquet", path_parquet)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", float(study.best_value))
        model_fp = os.path.join(cache_dir, f"tuned_{model_name}_{Path(path_parquet).stem}.joblib")
        ensure_dir(model_fp)
        joblib.dump(final_pipe, model_fp)
        mlflow.log_artifact(model_fp, artifact_path="models")

    return study, final_pipe

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="random_forest", help="random_forest | logreg | xgboost | lightgbm")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    study, model = tune_model(args.input, args.model, target_col=args.target, n_trials=args.trials, cache_dir=args.cache_dir)
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)