import os
import json
import argparse
from pathlib import Path
import importlib
import joblib

import numpy as np
import pandas as pd
import mlflow

# local modules (assume in project root)
eval_df_mod = importlib.import_module("evaluate_dataframe_potential")
eval_algo_mod = importlib.import_module("evaluate_algorithm_potential")
tuning_mod = importlib.import_module("model_hyperparam_tuning")  # exposes tune_model

def save_balanced_versions(df_fp, target_col, cache_dir, random_state=42):
    df = pd.read_parquet(df_fp, engine="pyarrow")
    balances = {"initial": None, "25_75": 0.25, "50_50": 0.5}
    out_paths = {}
    os.makedirs(cache_dir, exist_ok=True)
    for name, frac in balances.items():
        outp = os.path.join(cache_dir, f"{Path(df_fp).stem}_{name}.parquet")
        if frac is None:
            df.to_parquet(outp, engine='pyarrow')
        else:
            df_bal = eval_df_mod.resample_for_balance(df, target_col=target_col, positive_fraction=frac, random_state=random_state)
            df_bal.to_parquet(outp, engine='pyarrow')
        out_paths[name] = outp
    return out_paths

def pick_best_balance(eval_results):
    best_name = "initial"
    best_score = -1.0
    if not isinstance(eval_results, dict):
        return best_name
    for name, metrics in eval_results.items():
        try:
            score = float(metrics.get("test_auc", -1.0))
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_name = name
    return best_name

def evaluate_and_select(processed_parquet, target_col, cache_dir, test_size, random_state):
    df_eval = eval_df_mod.evaluate_dataframe(processed_parquet, target_col=target_col, test_size=test_size, random_state=random_state, cache_dir=cache_dir)
    balanced_paths = save_balanced_versions(processed_parquet, target_col, cache_dir, random_state=random_state)
    best_balance = pick_best_balance(df_eval)
    return df_eval, balanced_paths, best_balance

def evaluate_algorithms_on(path_parquet, target_col, test_size, random_state, cache_dir):
    return eval_algo_mod.evaluate_algorithms(path_parquet, target_col=target_col, test_size=test_size, random_state=random_state, cache_dir=cache_dir)

def tune_selected_models(path_parquet, target_col, models_to_tune, trials, random_state, cache_dir):
    results = {}
    for model_name in models_to_tune:
        study, fitted = tuning_mod.tune_model(path_parquet, model_name, target_col=target_col, n_trials=trials, random_state=random_state, cache_dir=cache_dir)
        results[model_name] = {"best_value": float(study.best_value), "best_params": study.best_params}
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline runner (assumes processed df_global already exists).")
    parser.add_argument("--input-parquet", dest="input_parquet", required=True, help="Input processed parquet (df_global / processed)")
    parser.add_argument("--cache-dir", dest="cache_dir", default=os.path.join(os.getcwd(), "cache"))
    parser.add_argument("--target-col", dest="target_col", default="TARGET")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    parser.add_argument("--trials", dest="trials", type=int, default=30)
    parser.add_argument("--tune-models", nargs="*", help="Models to tune (logreg random_forest xgboost lightgbm). If omitted, auto-select best from evaluate_algorithm_potential.")
    parser.add_argument("--mlflow-experiment", dest="mlflow_experiment", default="credit_scoring_pipeline_run")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    processed_fp = args.input_parquet
    if not os.path.exists(processed_fp):
        raise FileNotFoundError(f"Processed parquet not found: {processed_fp}")

    # Step: evaluate dataframe balances and pick best one
    df_eval, balanced_paths, best_balance = evaluate_and_select(processed_fp, args.target_col, args.cache_dir, args.test_size, args.random_state)

    # Step: evaluate algorithms on chosen best balance
    chosen_fp = balanced_paths.get(best_balance, processed_fp)
    algo_results = evaluate_algorithms_on(chosen_fp, args.target_col, args.test_size, args.random_state, cache_dir=args.cache_dir)

    # Decide which models to tune
    if args.tune_models:
        to_tune = args.tune_models
    else:
        # pick best model by test_auc
        best_model = None
        best_auc = -1.0
        for m, stats in (algo_results or {}).items():
            try:
                auc = float(stats.get("test_auc", -1.0))
            except Exception:
                auc = -1.0
            if auc > best_auc:
                best_auc = auc
                best_model = m
        mapping = {"logreg": "logreg", "random_forest": "random_forest", "gradient_boosting": None, "xgboost": "xgboost"}
        to_tune = [mapping.get(best_model, "random_forest")]

    # Filter unsupported names
    available = {"logreg", "random_forest"}
    try:
        import xgboost  # noqa
        available.add("xgboost")
    except Exception:
        pass
    try:
        import lightgbm  # noqa
        available.add("lightgbm")
    except Exception:
        pass

    to_tune = [m for m in (to_tune or []) if m and m in available]
    if not to_tune:
        to_tune = ["random_forest"]

    tuning_summary = tune_selected_models(chosen_fp, args.target_col, to_tune, args.trials, args.random_state, args.cache_dir)

    # Final summary in MLflow
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"pipeline_{Path(processed_fp).stem}"):
        mlflow.log_param("processed_parquet", processed_fp)
        mlflow.log_param("selected_balance", best_balance)
        mlflow.log_param("cache_dir", args.cache_dir)
        mlflow.log_param("target_col", args.target_col)
        mlflow.log_param("algo_results", json.dumps(algo_results or {}))
        mlflow.log_param("df_eval", json.dumps(df_eval or {}))
        mlflow.log_param("tuning_summary", json.dumps(tuning_summary or {}))

    print("Pipeline finished.")
    print("Processed:", processed_fp)
    print("Best balance:", best_balance)
    print("Algorithm results:", json.dumps(algo_results or {}, indent=2))
    print("Tuning summary:", json.dumps(tuning_summary or {}, indent=2))

if __name__ == "__main__":
    main()