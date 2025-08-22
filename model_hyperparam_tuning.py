import os
from pathlib import Path
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import mlflow
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix

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
        C = params.get("C", 1.0)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=1000))])
        return pipe
    if name == "random_forest":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=random_state, **params))])
        return pipe
    if name == "xgboost" and XGB_AVAILABLE:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss", **params))])
        return pipe
    if name == "lightgbm" and LGB_AVAILABLE:
        # note: lgb.LGBMClassifier accepts similar params
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", lgb.LGBMClassifier(random_state=random_state, **params))])
        return pipe
    raise ValueError(f"Unsupported model or missing dependency: {name}")


def _compute_custom_and_normalized(y_true, y_pred_bin, pos_proportion):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    custom = 2 * tp + 1 * tn - 1 * fp - 10 * fn
    n = len(y_true)
    pos_prop = max(pos_proportion, 1e-9)
    normalized = custom * (1.0 / max(1, n)) * (1.0 / pos_prop)
    return float(custom), float(normalized)


def _best_threshold_max_recall(y_true, y_pred_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_recall = -1.0
    for t in thresholds:
        r = recall_score(y_true, (y_pred_proba >= t).astype(int))
        if r > best_recall:
            best_recall = r
            best_t = t
    return best_t, best_recall


def validate_tune_inputs(path_parquet: str, model_name: str, n_trials: int, cache_dir: str):
    errs = []
    if not path_parquet or not os.path.exists(path_parquet):
        errs.append(f"input parquet not found: {path_parquet}")
    if n_trials is None or n_trials <= 0:
        errs.append(f"trials must be > 0, got: {n_trials}")
    allowed = {"random_forest", "logreg"}
    if XGB_AVAILABLE:
        allowed.add("xgboost")
    if LGB_AVAILABLE:
        allowed.add("lightgbm")
    if model_name not in allowed:
        errs.append(f"model_name '{model_name}' not supported by installed backends. Allowed: {sorted(allowed)}")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        errs.append(f"cache_dir not creatable: {cache_dir} -> {e}")
    if errs:
        msg = "tune_model input validation failed:\n  " + "\n  ".join(errs)
        print(f"[tune_model][VALIDATION] {msg}")
        raise ValueError(msg)
    print(f"[tune_model][VALIDATION] OK: parquet={path_parquet}, model={model_name}, trials={n_trials}, cache={cache_dir}")


def _load_df(df_or_path):
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    return pd.read_parquet(df_or_path, engine="pyarrow")


def tune_model(df_or_path: str, model_name: str, target_col: str = "TARGET",
               n_trials: int = 50, random_state: int = 42, cache_dir: str = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\cache",
               persist_parquet: bool = False):
    validate_tune_inputs(df_or_path if not isinstance(df_or_path, pd.DataFrame) else None, model_name, n_trials, cache_dir)
    df = _load_df(df_or_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("No numeric features available for tuning")
    y = df[target_col].astype(int)
    pos_prop_global = float(y.mean()) if len(y) > 0 else 0.0

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def _cv_custom_score(pipe):
        customs = []
        normals = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            p_val = pipe.predict_proba(X_val)[:, 1]
            best_t, _ = _best_threshold_max_recall(y_val, p_val)
            y_pred_bin = (p_val >= best_t).astype(int)
            custom, normalized = _compute_custom_and_normalized(y_val, y_pred_bin, pos_prop_global)
            customs.append(custom)
            normals.append(normalized)
        return float(np.mean(customs)), float(np.mean(normals))

    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        pipe = make_pipeline_for("random_forest", params, random_state)
        mean_custom, _ = _cv_custom_score(pipe)
        return mean_custom

    def objective_logreg(trial):
        C = trial.suggest_loguniform("C", 1e-4, 1e2)
        params = {"C": C}
        pipe = make_pipeline_for("logreg", params, random_state)
        mean_custom, _ = _cv_custom_score(pipe)
        return mean_custom

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        pipe = make_pipeline_for("xgboost", params, random_state)
        mean_custom, _ = _cv_custom_score(pipe)
        return mean_custom

    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        pipe = make_pipeline_for("lightgbm", params, random_state)
        mean_custom, _ = _cv_custom_score(pipe)
        return mean_custom

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

    # compute best threshold on full dataset (maximize recall) and compute normalized score on full data
    p_all = final_pipe.predict_proba(X)[:, 1]
    best_t_full, best_recall_full = _best_threshold_max_recall(y, p_all)
    y_pred_bin_full = (p_all >= best_t_full).astype(int)
    final_custom, final_normalized = _compute_custom_and_normalized(y, y_pred_bin_full, pos_prop_global)

    # Log to MLflow (store mlruns under cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        mlflow.set_tracking_uri(f"file://{os.path.join(os.path.abspath(cache_dir), 'mlruns')}")
    except Exception:
        pass
    mlflow.set_experiment("credit_scoring_hyperparam_tuning")
    run_name = f"tune_{model_name}_{Path(path_parquet).stem}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_parquet", path_parquet)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_custom_score", float(study.best_value))
        mlflow.log_metric("final_custom_score", float(final_custom))
        mlflow.log_metric("final_normalized_custom", float(final_normalized))
        mlflow.log_metric("final_best_threshold", float(best_t_full))
        mlflow.log_param("pos_proportion", float(pos_prop_global))
        model_fp = os.path.join(cache_dir, f"tuned_{model_name}_{Path(path_parquet).stem}.joblib")
        ensure_dir(model_fp)
        joblib.dump(final_pipe, model_fp)
        mlflow.log_artifact(model_fp, artifact_path="models")

    try:
        final_pipe.best_threshold_ = float(best_t_full)
    except Exception:
        pass

    return study, final_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet to tune on")
    parser.add_argument("--model", required=True, choices=["logreg", "random_forest", "xgboost", "lightgbm"], help="Model to tune")
    parser.add_argument("--target", default="TARGET")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    study, pipe = tune_model(args.input, args.model, target_col=args.target, n_trials=args.trials, random_state=args.random_state, cache_dir=args.cache_dir)
    print("Best params:", study.best_params)
    print("Best value (cv custom):", study.best_value)