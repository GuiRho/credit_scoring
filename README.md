This repository contains a credit-scoring pipeline:
- evaluate_dataframe_potential.py — evaluate different target balances (initial, 25/75, 50/50)
- evaluate_algorithm_potential.py — compare classifiers on a chosen dataframe
- model_hyperparam_tuning.py — Optuna tuning for multiple model families
- run_pipeline.py — orchestrator (assumes processed dataframe already exists)
- processing/process_df_global.py — feature engineering / processing (kept for offline use)

Installation (Windows PowerShell / CMD)
1. Create and activate a virtual environment (recommended)
   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies
   python -m pip install --upgrade pip
   python -m pip install pandas numpy scikit-learn mlflow optuna joblib pyarrow xgboost lightgbm

Run examples
- Build dataframes 

- Evaluate dataframes (produces MLflow runs and returns metrics)
  python evaluate_dataframe_potential.py --input path\to\processed.parquet --target TARGET --cache-dir cache

- Evaluate algorithms on a processed parquet
  python evaluate_algorithm_potential.py --input path\to\processed.parquet --target TARGET --cache-dir cache

- Tune a model with Optuna (example: random_forest)
  python model_hyperparam_tuning.py --input path\to\processed.parquet --model random_forest --trials 50 --cache-dir cache

- Full orchestrated run (assumes the parquet passed is already processed)
  python run_pipeline.py --input-parquet path\to\processed.parquet --cache-dir cache --trials 30

Notes
- MLflow: by default logs to local mlruns/ folder. To use a remote tracking server set MLflow environment variables (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, etc.) before running.
- The orchestrator run_pipeline.py intentionally skips raw ingestion/one-time preprocessing — provide the processed df_global parquet as input.