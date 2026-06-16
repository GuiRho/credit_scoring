# Credit Scoring

End-to-end credit scoring ML pipeline — data ingestion through model serving.

Cleaned and restructured by **superclean** (v2). See `v1/history/` for originals.

## Structure

```
credit_scoring/
├── src/credit_scoring/       # Package source
│   ├── ingest.py             # Data ingestion (joins 6 Home Credit sources)
│   ├── preprocess.py         # Cleaning, imputation, outlier removal
│   ├── features.py           # Feature selection and engineering
│   ├── balance.py            # Balancing strategies (SMOTE, undersampling)
│   ├── algo_choice.py        # Algorithm comparison (LR/RF/GBM/XGB/CatBoost)
│   ├── tuning.py             # Hyperparameter search (GridSearchCV / Optuna)
│   ├── analysis.py           # SHAP analysis (local + global importance)
│   ├── plots.py              # SHAP visualization helpers
│   ├── drift.py              # Evidently data drift monitoring
│   ├── serving.py            # FastAPI prediction API
│   ├── dashboard.py          # Streamlit dashboard
│   ├── models.py             # Model loading utilities
│   └── config.py             # Unified configuration
├── tests/                    # Pytest test suite
├── config/                   # Pipeline configuration JSONs
├── data/                     # Data directory (gitignored)
├── production_model/         # Deployed model artifact (in v1/history/)
├── v1/history/               # Original pre-cleanup files
├── pyproject.toml            # Package configuration
├── sample_data.json          # Example client features
├── probable_default.json     # High-risk example
└── improbable_default.json   # Low-risk example
```

## Quick Start

```bash
pip install -e .              # Install package
uvicorn credit_scoring.serving:app  # Start serving API
streamlit run credit_scoring/dashboard.py  # Start dashboard
pytest tests/                 # Run tests
```

## Pipeline

1. `ingest.py` → loads raw CSVs, joins 6 data sources
2. `preprocess.py` → cleans, imputes, removes outliers
3. `features.py` → selects + engineers features
4. `balance.py` → tests oversampling/undersampling strategies
5. `algo_choice.py` → compares ML algorithms
6. `tuning.py` → tunes hyperparameters
7. `analysis.py` → SHAP analysis (optional, post-training)

All steps log to MLflow. See `v1/history/README.md` for original documentation.
