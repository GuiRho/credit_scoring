import os
import json
import argparse
import importlib
from pathlib import Path
import mlflow
import pandas as pd

# local modules (assume in project root)
eval_df_mod = importlib.import_module("evaluate_dataframe_potential")
eval_algo_mod = importlib.import_module("evaluate_algorithm_potential")
tuning_mod = importlib.import_module("model_hyperparam_tuning")
process_df_mod = importlib.import_module("process_df_global")

def save_balanced_versions(df_or_path, target_col, cache_dir, random_state=42, persist_parquet: bool = False):
    """
    Returns mapping: balance_name -> DataFrame (and optionally writes parquets if persist_parquet True).
    """
    df = df_or_path if isinstance(df_or_path, pd.DataFrame) else pd.read_parquet(df_or_path, engine="pyarrow")
    balances = {"initial": None, "25_75": 0.25, "50_50": 0.5}
    out = {}
    for name, frac in balances.items():
        if frac is None:
            df_bal = df.copy()
        else:
            df_bal = eval_df_mod.resample_for_balance(df, target_col=target_col, positive_fraction=frac, random_state=random_state)
        fp = None
        if persist_parquet:
            stem = Path(getattr(df_or_path, 'name', 'df')).stem if isinstance(df_or_path, str) else 'df'
            fp = os.path.join(cache_dir, f"balanced_{stem}_{name}.parquet")
            df_bal.to_parquet(fp, engine="pyarrow")
        out[name] = {"df": df_bal, "parquet": fp}
    return out

def pick_best_balance(eval_results):
    best_name = "initial"
    best_score = float("-inf")
    if not isinstance(eval_results, dict):
        return best_name
    for name, metrics in eval_results.items():
        if not isinstance(metrics, dict):
            continue
        # prefer normalized custom score if available, otherwise fallback to test_auc
        val = None
        if "test_normalized_custom" in metrics:
            try:
                val = float(metrics["test_normalized_custom"])
            except (ValueError, TypeError):
                val = None
        if val is None and "test_auc" in metrics:
            try:
                val = float(metrics["test_auc"])
            except (ValueError, TypeError):
                val = None
        if val is None:
            continue
        if val > best_score:
            best_score = val
            best_name = name
    return best_name

def evaluate_and_select(processed_parquet, target_col, cache_dir, test_size, random_state):
    df_eval_results = eval_df_mod.evaluate_dataframe(processed_parquet, target_col=target_col, test_size=test_size, random_state=random_state, cache_dir=cache_dir)
    balanced_paths = save_balanced_versions(processed_parquet, target_col, cache_dir, random_state=random_state)
    best_balance = pick_best_balance(df_eval_results.get("metrics", {}))
    chosen_fp = balanced_paths.get(best_balance, {}).get("parquet", processed_parquet)
    return {"df_eval": df_eval_results, "balanced_paths": balanced_paths, "best_balance": best_balance, "chosen_fp": chosen_fp}

def evaluate_algorithms_on(path_parquet, target_col, test_size, random_state, cache_dir):
    return eval_algo_mod.evaluate_algorithms(path_parquet, target_col=target_col, test_size=test_size, random_state=random_state, cache_dir=cache_dir)

def tune_selected_models(df_or_path, target_col, models_to_tune, trials, random_state, cache_dir):
    results = {}
    for model_name in models_to_tune:
        try:
            study, pipe = tuning_mod.tune_model(df_or_path, model_name, target_col=target_col, n_trials=trials, random_state=random_state, cache_dir=cache_dir)
            results[model_name] = {
                "best_params": study.best_params,
                "best_cv_custom": float(study.best_value),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline runner driven by pipeline_config.json (default).")
    default_cfg = os.path.join(os.getcwd(), "pipeline_config.json")
    parser.add_argument("--pipeline-config", dest="pipeline_config", default=default_cfg,
                        help=f"Path to pipeline_config.json (default: {default_cfg}). The pipeline_config.json is the single source of truth.")
    parser.add_argument("--cache-dir", dest="cache_dir", default=r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\cache",
                        help="Override cache dir (not recommended). By default artifacts are stored outside the repo.")
    parser.add_argument("--mlflow-experiment", dest="mlflow_experiment", default="credit_scoring_pipeline_run")
    return parser.parse_args()

def validate_pipeline_config(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("pipeline_config must contain a non-empty 'entries' list")
    missing = []
    for i, e in enumerate(entries, start=1):
        if "input_parquet" not in e:
            missing.append((i, "input_parquet"))
        if "output_parquet" not in e:
            missing.append((i, "output_parquet"))
    if missing:
        msgs = [f"entry #{idx} missing key: {k}" for idx, k in missing]
        raise ValueError("Invalid pipeline_config entries:\n  " + "\n  ".join(msgs))
    print(f"[run_pipeline][VALIDATION] pipeline_config OK: {len(entries)} entries.")

def configure_mlflow(cache_dir: str, experiment_name: str):
    """Central MLflow configuration used by the pipeline (creates experiment if missing)."""
    os.makedirs(cache_dir, exist_ok=True)
    mlruns_dir = os.path.join(os.path.abspath(cache_dir), "mlruns")
    try:
        # Path.as_uri() produces a valid file:// URI on Windows
        mlflow_uri = Path(mlruns_dir).as_uri()
        mlflow.set_tracking_uri(mlflow_uri)
    except Exception:
        # fallback to plain absolute path with file:// prefix
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    # create or get experiment
    mlflow.set_experiment(experiment_name)
    print(f"[run_pipeline] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    exp = mlflow.get_experiment_by_name(experiment_name)
    print(f"[run_pipeline] MLflow experiment: name='{experiment_name}', id={exp.experiment_id if exp else 'NOT_CREATED'}")
    # also export env var so subprocesses/libraries reuse it
    os.environ["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()

def main():
    args = parse_args()

    configure_mlflow(args.cache_dir, args.mlflow_experiment)

    if not os.path.exists(args.pipeline_config):
        raise FileNotFoundError(f"pipeline_config.json not found: {args.pipeline_config}")
    with open(args.pipeline_config, "r", encoding="utf8") as f:
        cfg = json.load(f)
        
    if isinstance(cfg, dict) and "entries" in cfg:
        entries = cfg["entries"]
    elif isinstance(cfg, list):
        entries = cfg
    else:
        raise ValueError("pipeline_config must be a list or a dict with an 'entries' list")

    validate_pipeline_config(entries)
    print(f"[run_pipeline] Loaded pipeline_config with {len(entries)} entries from {args.pipeline_config}")

    cfg_objs = []
    for i, ent in enumerate(entries, start=1):
        print(f"[run_pipeline] (phase1) [{i}/{len(entries)}] Processing entry via process_df_global")
        cfg_fields = set(getattr(process_df_mod.Config, "__annotations__", {}).keys())
        cfg_kwargs = {k: v for k, v in ent.items() if k in cfg_fields}
        ignored_keys = [k for k in ent.keys() if k not in cfg_fields]
        if ignored_keys:
            print(f"[run_pipeline] Note: ignoring non-Config keys for process_df_global: {ignored_keys}")
        cfg_obj = process_df_mod.Config(**cfg_kwargs)
        print(f"[run_pipeline]       input_parquet={cfg_obj.input_parquet} -> output_parquet={cfg_obj.output_parquet}")
        process_df_mod.main_cfg(cfg_obj)
        cfg_objs.append(cfg_obj)

    all_balanced_store = {}
    for cfg_obj in cfg_objs:
        processed_fp = cfg_obj.output_parquet
        tgt = getattr(cfg_obj, "target_col", "TARGET")
        cache = getattr(cfg_obj, "cache_dir", args.cache_dir) or args.cache_dir
        os.makedirs(cache, exist_ok=True)
        print(f"[run_pipeline] (phase2) Saving balanced versions for processed file: {processed_fp} -> cache {cache}")
        if not os.path.exists(processed_fp):
            raise FileNotFoundError(f"Expected processed output not found: {processed_fp}")
        processed_df = pd.read_parquet(processed_fp, engine="pyarrow")
        balanced_store = save_balanced_versions(processed_df, tgt, cache, random_state=getattr(cfg_obj, "random_state", 42), persist_parquet=getattr(cfg_obj, "persist_parquet", False))
        all_balanced_store[processed_fp] = balanced_store

    overall_results = {}
    for ent in entries:
        cfg_fields = set(getattr(process_df_mod.Config, "__annotations__", {}).keys())
        cfg_kwargs = {k: v for k, v in ent.items() if k in cfg_fields}
        cfg_obj = process_df_mod.Config(**cfg_kwargs)

        processed_fp = cfg_obj.output_parquet
        cache = ent.get("cache_dir", args.cache_dir) or args.cache_dir
        test_size = ent.get("test_size", 0.2)
        random_state = ent.get("random_state", 42)
        trials = ent.get("trials", 30)
        print(f"[run_pipeline] (phase3) Evaluating processed file: {processed_fp}")

        df_eval_results = eval_df_mod.evaluate_dataframe(processed_fp, target_col=cfg_obj.target_col, test_size=test_size, random_state=random_state, cache_dir=cache, persist_parquet=ent.get("persist_parquet", False))
        eval_metrics = df_eval_results.get("metrics") if isinstance(df_eval_results, dict) else df_eval_results

        balanced_store = all_balanced_store.get(processed_fp, {})
        best_balance = pick_best_balance(eval_metrics)
        chosen_entry = balanced_store.get(best_balance)
        
        if chosen_entry and "df" in chosen_entry:
            chosen_df = chosen_entry["df"]
        else:
            chosen_df = pd.read_parquet(processed_fp, engine="pyarrow")

        print(f"[run_pipeline] Selected best balance '{best_balance}' (rows={len(chosen_df)})")

        algo_results = eval_algo_mod.evaluate_algorithms(chosen_df, target_col=cfg_obj.target_col, test_size=test_size, random_state=random_state, cache_dir=cache)

        to_tune = ent.get("tune_models")
        if not to_tune:
            best_model = None
            best_val = float("-inf")
            for m, stats in (algo_results or {}).items():
                val = float(stats.get("test_normalized_custom", stats.get("test_auc", -1.0)))
                if val > best_val:
                    best_val = val
                    best_model = m
            to_tune = [best_model] if best_model else ["random_forest"]

        available = {"logreg", "random_forest"}
        try:
            import xgboost
            available.add("xgboost")
        except ImportError:
            pass
        try:
            import lightgbm
            available.add("lightgbm")
        except ImportError:
            pass
            
        to_tune = [m for m in (to_tune or []) if m and m in available]
        if not to_tune:
            to_tune = ["random_forest"]

        tuning_summary = tune_selected_models(chosen_df, cfg_obj.target_col, to_tune, trials, random_state, cache)
        overall_results[processed_fp] = {"df_eval": df_eval_results, "algo_results": algo_results, "tuning_summary": tuning_summary}
        print(f"[run_pipeline] Completed entry for {processed_fp}")

    print("[run_pipeline] Pipeline finished for all entries.")
    out_summary = os.path.join(args.cache_dir, "pipeline_overall_results.json")
    with open(out_summary, "w", encoding="utf8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"[run_pipeline] Summary saved to {out_summary}")

if __name__ == "__main__":
    main()