# Credit Scoring Project

GitHub :https://github.com/GuiRho/credit_scoring

## Project Goal
This project aims to develop, deploy, and monitor a credit scoring model. It encompasses the entire machine learning lifecycle, from data ingestion and model building to API serving, data drift analysis, and a user-facing dashboard.

## Project Structure and Directories

-   `.github/`: Contains GitHub Actions workflows for continuous integration and deployment (e.g., `deploy.yml`).
-   `.pytest_cache/`: Pytest cache directory, used to speed up test runs.
-   `catboost_info/`: Stores information and logs generated during CatBoost model training.
-   `data_drift_analysis/`: Contains scripts and reports for analyzing data drift (e.g., `drift.py`, `data_drift_report.html`).
-   `mlruns/`: MLflow tracking server directory, used to log parameters, metrics, and artifacts from model training runs.
-   `model_analysis/`: Scripts and results for in-depth model analysis, including SHAP plots (e.g., `analysis.py`, `analysis_results/`).
-   `model_building/`: Core scripts for building the credit scoring model, including data preprocessing, feature engineering, and model training/tuning.
    -   `algo_choice.py`: Logic for algorithm selection.
    -   `balance.py`: Scripts related to data balancing techniques.
    -   `ingest.py`: Handles data ingestion.
    -   `preprocess.py`: Data preprocessing steps.
    -   `process.py`: Feature processing.
    -   `tuning.py`: Model hyperparameter tuning.
-   `model_serving/`: Contains the FastAPI application for serving the trained model as a REST API.
    -   `docker_main.py`: The main entry point for the Dockerized API.
    -   `main.py`: Likely a development version of the API.
    -   `Dockerfile`: Dockerfile for containerizing the API.
-   `production_model/`: Stores the final, production-ready model artifact and related metadata.
    -   `model.pkl`: The serialized machine learning model.
    -   `conda.yaml`, `python_env.yaml`: Conda/Python environment definitions for reproducibility.
    -   `input_example.json`, `serving_input_example.json`: Example input data for the model and serving API.
-   `tests/`: Contains unit and integration tests for various components of the project.
    -   `conftest.py`: Pytest fixtures for shared test setup.
    -   `test_analysis.py`: Tests for model analysis.
    -   `test_drift.py`: Tests for data drift analysis.
    -   `test_dashboard_api.py`: Tests for the model serving API.
    -   `test_model_scoring.py`: Tests for the core model's prediction logic.
    -   `test_robustness.py`: Tests for input validation and error handling.
-   `UX/`: Contains the user interface (dashboard) application.
    -   `app.py`: The main dashboard application.
    -   `Dockerfile`: Dockerfile for containerizing the dashboard.
-   `package_model.py`: Script for packaging the trained model.
-   `run_tests.py`: Script to execute the test suite.
