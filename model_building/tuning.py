import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

import optuna
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.base import BaseEstimator

from mlflow.models.signature import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Optional Imports for Advanced Models ---
XGB_AVAILABLE = False
LGB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. XGBClassifier will not be used.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. LGBMClassifier will not be used.")


def _compute_custom_and_normalized(y_true: np.ndarray, y_pred_bin: np.ndarray, 
                                  pos_proportion: float) -> Tuple[float, float]:
    """Compute custom and normalized scores based on confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    custom = (2 * tp) + (1 * tn) - (1 * fp) - (10 * fn)
    n = len(y_true)
    pos_prop = max(pos_proportion, 1e-9)
    normalized = custom * (1.0 / max(1, n)) * (1.0 / pos_prop)
    return float(custom), float(normalized)


def _find_best_threshold_custom_score(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                     pos_proportion: float) -> Tuple[float, float]:
    """Find the best threshold that maximizes the custom normalized score."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -np.inf
    
    for t in thresholds:
        y_pred_bin = (y_pred_proba >= t).astype(int)
        _, norm_score = _compute_custom_and_normalized(y_true, y_pred_bin, pos_proportion)
        if norm_score > best_score:
            best_score = norm_score
            best_t = t
            
    return best_t, best_score





def _run_grid_search_logreg(X_train: pd.DataFrame, y_train: pd.Series, 
                           pos_prop_global: float, random_state: int) -> Tuple[Pipeline, Dict[str, Any], float]:
    """Run GridSearchCV for Logistic Regression with custom scoring."""
    logger.info("Starting GridSearchCV for Logistic Regression...")
    
    def custom_scorer_func(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        best_t, _ = _find_best_threshold_custom_score(y_true, y_pred_proba, pos_prop_global)
        y_pred_bin = (y_pred_proba >= best_t).astype(int)
        _, normalized_score = _compute_custom_and_normalized(y_true, y_pred_bin, pos_prop_global)
        return normalized_score

    custom_scorer = make_scorer(custom_scorer_func, needs_proba=True, greater_is_better=True)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=random_state))
    ])
    
    param_grid = [
        {
            'clf__solver': ['liblinear'],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__class_weight': [None, 'balanced']
        },
        {
            'clf__solver': ['saga'],
            'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__class_weight': [None, 'balanced'],
            'clf__l1_ratio': np.linspace(0, 1, 5)
        },
        {
            'clf__solver': ['lbfgs'],
            'clf__penalty': ['l2', 'none'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__class_weight': [None, 'balanced']
        }
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=custom_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}
    
    return grid_search.best_estimator_, best_params, grid_search.best_score_


def get_optuna_suggest_and_model(model_name: str, random_state: int) -> Tuple[Callable, BaseEstimator]:
    """Return appropriate parameter suggestion function and model instance for Optuna."""
    def suggest_params_rf(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4)
        }
    
    def suggest_params_xgb(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }
    
    def suggest_params_lgb(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("num_leaves", 10, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }
    
    if model_name == "random_forest":
        return suggest_params_rf, RandomForestClassifier(random_state=random_state, n_jobs=-1)
    elif model_name == "xgboost" and XGB_AVAILABLE:
        return suggest_params_xgb, XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
    elif model_name == "lightgbm" and LGB_AVAILABLE:
        return suggest_params_lgb, lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported or unavailable model: {model_name}")


def tune_and_log_model(config: Dict[str, Any], cache_dir: str, random_state: int, register_as: Optional[str] = None) -> None:
    """Main function to tune and log model using either GridSearchCV or Optuna."""
    # Load and prepare data
    input_dir, model_name, n_trials, target_col = config['dataset_dir'], config['model_name'], config.get('n_trials'), 'TARGET'
    
    train_path = os.path.join(input_dir, "train_processed.parquet")
    test_path = os.path.join(input_dir, "test_processed.parquet")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train parquet not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test parquet not found at {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    if target_col not in df_train.columns or target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    X_train = df_train.drop(columns=[target_col]).select_dtypes(include=np.number)
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col]).select_dtypes(include=np.number)
    y_test = df_test[target_col]

    # --- ROBUSTNESS FIX: Convert integer columns to float64 ---
    int_cols = X_train.select_dtypes(include=['int32', 'int64']).columns
    if len(int_cols) > 0:
        logger.info(f"Converting {len(int_cols)} integer columns to float64 to prevent schema errors.")
        X_train[int_cols] = X_train[int_cols].astype('float64')
        X_test[int_cols] = X_test[int_cols].astype('float64')
    # -----------------------------------------------------------
    
    pos_prop_global = float(y_train.mean())

    if model_name == 'logreg':
        # ... (GridSearch logic remains the same)
        final_pipeline, best_params, best_cv_score = _run_grid_search_logreg(
            X_train, y_train, pos_prop_global, random_state
        )
    else:
        # ... (Optuna logic remains the same)
        if not n_trials:
            raise ValueError("'n_trials' must be set in config for Optuna models.")
        
        suggest_params, base_model = get_optuna_suggest_and_model(model_name, random_state)
        
        def objective(trial: optuna.Trial) -> float:
            # (Objective function is unchanged)
            params = suggest_params(trial)
            try:
                model = base_model.set_params(**params)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                scores = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                    pipe.fit(X_tr, y_tr)
                    p_val = pipe.predict_proba(X_val)[:, 1]
                    
                    best_t, _ = _find_best_threshold_custom_score(y_val, p_val, pos_prop_global)
                    _, normalized_score = _compute_custom_and_normalized(
                        y_val, (p_val >= best_t).astype(int), pos_prop_global
                    )
                    scores.append(normalized_score)
                
                return np.mean(scores)
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed with error: {e}")
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params
        best_cv_score = study.best_value
        final_model = base_model.set_params(**best_params)
        final_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", final_model)])
        final_pipeline.fit(X_train, y_train)

    # Evaluate final model on test set
    p_train_final = final_pipeline.predict_proba(X_train)[:, 1]
    best_threshold, _ = _find_best_threshold_custom_score(y_train, p_train_final, pos_prop_global)
    
    p_test = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred_bin = (p_test >= best_threshold).astype(int)
    
    test_custom, test_normalized = _compute_custom_and_normalized(y_test, y_test_pred_bin, pos_prop_global)
    cm = confusion_matrix(y_test, y_test_pred_bin)
    
    # Log to MLflow
    run_name = f"tune_{model_name}_{Path(input_dir).name}"
    
    # --- HERE IS THE FIX ---
    with mlflow.start_run(run_name=run_name) as run: # <-- Added 'as run'
        mlflow.log_params(config)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({
            "best_cv_normalized_score": best_cv_score,
            "test_normalized_custom_score": test_normalized,
            "test_custom_score": test_custom,
            "best_threshold": best_threshold,
        })
        mlflow.log_param("best_threshold", best_threshold)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix (Test Set)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        input_example = X_train.head(5)
        signature = infer_signature(input_example, final_pipeline.predict_proba(input_example))
        
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            metadata={"best_threshold": float(best_threshold)}
        )

        # Now this part will work correctly
        if register_as:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Registering model '{register_as}' from URI: {model_uri}")
            # Use registered_model_name for clarity
            mlflow.register_model(model_uri=model_uri, name=register_as)
            logger.info("Model registration complete.")
        # --- END OF NEW CODE ---
    
    logger.info(f"\n{'='*20} Tuning Complete {'='*20}")
    logger.info(f"Tuning method: {'GridSearchCV' if model_name == 'logreg' else 'Optuna'}")
    logger.info(f"Best CV Score: {best_cv_score:.4f}")
    logger.info(f"Test Normalized Score: {test_normalized:.4f}")
    logger.info(f"Best Threshold: {best_threshold:.3f}")
    logger.info(f"Check MLflow experiment 'Hyperparameter Tuning' for run '{run_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a model's hyperparameters using Optuna or GridSearchCV")
    parser.add_argument("--config", required=True, help="Path to the tuning_config.json file.")
    parser.add_argument("--cache-dir", default="C:/Users/gui/Documents/OpenClassrooms/Projet 7/cache", help="Directory for MLflow runs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--register-as", help="Register the best model with this name in the MLflow registry.")
    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    with open(args.config, 'r') as f:
        tuning_config = json.load(f)

    # Validate required config fields
    required_fields = ['dataset_dir', 'model_name']
    for field in required_fields:
        if field not in tuning_config:
            raise ValueError(f"Missing required field in config: {field}")

    # Set tracking URI to the desired cache directory
    tracking_uri = "file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Hyperparameter Tuning")
    print(f"MLflow is configured to track experiments to: {mlflow.get_tracking_uri()}")
    
    try:
        tune_and_log_model(
            config=tuning_config, 
            cache_dir=args.cache_dir, 
            random_state=args.random_state,
            register_as=args.register_as
        )
    except Exception as e:
        logger.error(f"Tuning failed with error: {e}")
        raise