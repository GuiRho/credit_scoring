"""
Model Analysis Script : Runs SHAP analysis on a production model.

This script expects:
1.  A saved MLflow model in the directory specified by the MODEL_DIR variable.
2.  The processed test dataset ('test_processed.parquet') to be located in the 
    directory specified by the DATASET_DIR variable.

If these prerequisites are not met, the script will exit with an error.
"""
import os
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# --- CONFIGURATION ---

# 1. Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define Paths Relative to Project Root
MODEL_DIR = PROJECT_ROOT / "production_model"
OUTPUT_DIR = PROJECT_ROOT / "model_analysis" / "analysis_results"
DATASET_DIR = PROJECT_ROOT

# 3. Analysis Parameters
SAMPLE_SIZE = 500
TOP_N_FEATURES = 8
PLOT_DPI = 150

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ANALYSIS FUNCTIONS ---

def load_packaged_model(model_path):
    """
    Loads the model and its metadata by parsing the MLmodel file.
    Returns the loaded model object.
    """
    model_path = Path(model_path)
    mlmodel_path = model_path / "MLmodel"

    if not mlmodel_path.is_file():
        logger.error(f"FATAL: Model not found. The file '{mlmodel_path.resolve()}' does not exist.")
        sys.exit(1)

    try:
        model = mlflow.pyfunc.load_model(str(model_path))
        with open(mlmodel_path, 'r') as f:
            mlmodel_data = yaml.safe_load(f)
        custom_metadata = mlmodel_data.get("metadata", {})
        best_threshold = float(custom_metadata.get('best_threshold', 0.5))
        
        logger.info("--- Packaged model loaded successfully ---")
        logger.info(f"Model path: {model_path.resolve()}")
        logger.info(f"Best Threshold: {best_threshold}")
        return model
    except Exception as e:
        logger.error(f"FATAL: An unexpected error occurred while loading the model: {e}", exc_info=True)
        sys.exit(1)

def load_test_data(data_dir):
    """Loads the processed test data from the specified directory."""
    try:
        logger.info(f"Attempting to load test data from: {data_dir}")
        test_path = Path(data_dir) / "test_processed.parquet"
        if not test_path.is_file():
            raise FileNotFoundError(f"Ensure 'test_processed.parquet' exists in '{data_dir}'")
        test_df = pd.read_parquet(test_path)
        X_test, y_test = test_df.drop(columns=["TARGET"]), test_df["TARGET"]
        logger.info(f"Test data shape: {X_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"FATAL: Could not load data: {e}", exc_info=True)
        sys.exit(1)

def sample_data(X_test, y_test, sample_size):
    """Stratified sampling of the test data."""
    logger.info(f"Sampling {sample_size} rows from the test set...")
    try:
        actual_size = min(sample_size, len(X_test))
        if actual_size < sample_size:
            logger.warning(f"Requested sample size {sample_size} > test set size {len(X_test)}. Using {actual_size}.")
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=actual_size, random_state=42)
        _, sample_idx = next(strat_split.split(X_test, y_test))
        X_sample = X_test.iloc[sample_idx]
        y_sample = y_test.iloc[sample_idx]
        logger.info(f"Sample shape: {X_sample.shape}")
        return X_sample, y_sample
    except Exception as e:
        logger.error(f"Error during data sampling: {e}", exc_info=True)
        sys.exit(1)

def get_top_features_by_shap(shap_values, feature_names, n_features):
    """
    Determines top features based on the mean absolute SHAP value summed across all classes.
    """
    # Sum the absolute SHAP values across all classes for each feature
    # This gives a measure of total importance, regardless of class direction
    global_shap_importance = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    
    # Calculate the mean of these global importances across all samples
    mean_abs_shap = np.mean(global_shap_importance, axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance_df.head(n_features)['feature'].tolist()
    logger.info(f"Top {n_features} features (global importance across all classes): {top_features}")
    return top_features

def generate_summary_plots(shap_values, X_sample, class_names, output_dir):
    """Generates and saves SHAP summary plots for each class."""
    logger.info("Generating SHAP Summary Plots...")
    for i, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(shap_values[i], X_sample, max_display=20, show=False)
        plt.title(f'SHAP Summary Plot (Top 20 Features) - {class_name}', fontsize=16)
        plt.gcf().set_size_inches(12, 8)
        plt.tight_layout()
        path = output_dir / f"summary_plot_{class_name}.png"
        plt.savefig(path, dpi=PLOT_DPI)
        plt.close()
        logger.info(f"Saved summary plot to {path}")

def generate_dependency_plots(shap_values, X_sample, features, class_names, output_dir):
    """Generates and saves SHAP dependency plots for each of the top features, for each class."""
    logger.info("Generating SHAP Dependency Plots for all classes...")
    base_dep_plot_dir = output_dir / "dependency_plots"
    os.makedirs(base_dep_plot_dir, exist_ok=True)
    
    for class_idx, class_name in enumerate(class_names):
        class_dep_plot_dir = base_dep_plot_dir / class_name.replace(' ', '_')
        os.makedirs(class_dep_plot_dir, exist_ok=True)
        logger.info(f"Generating plots for {class_name} in '{class_dep_plot_dir}'")

        for feature in features:
            plt.figure()
            shap.dependence_plot(
                feature, 
                shap_values[class_idx], 
                X_sample, 
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {feature} (for {class_name})', fontsize=14)
            plt.tight_layout()
            sanitized_name = "".join(c for c in feature if c.isalnum() or c in ('_', '-')).rstrip()
            path = class_dep_plot_dir / f"dependency_{sanitized_name}.png"
            plt.savefig(path, dpi=PLOT_DPI)
            plt.close()
            logger.info(f"  Saved dependency plot for '{feature}'")


def save_global_shap_importance(shap_values, feature_names, class_names, model_dir):
    """
    Calculates and saves the global feature importance (mean absolute SHAP value)
    for the positive class.
    """
    logger.info("Calculating and saving global feature importance...")
    
    # Assuming class '1' (Default) is the positive class, which is typically the second one.
    positive_class_index = 1
    if len(class_names) < 2:
        logger.warning("Model does not have two classes. Defaulting to first class for SHAP importance.")
        positive_class_index = 0

    # Extract SHAP values for the positive class
    shap_values_positive_class = shap_values[positive_class_index]
    
    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values_positive_class).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    # Save to a file in the model directory so the dashboard can find it
    output_path = Path(model_dir) / "global_feature_importance.json"
    importance_df.to_json(output_path, orient="records")
    logger.info(f"Global feature importance saved to: {output_path}")


def main():
    """Main function to run the SHAP analysis."""
    logger.info("Starting SHAP analysis...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory for plots: {OUTPUT_DIR.resolve()}")

    model = load_packaged_model(MODEL_DIR)
    X_test, y_test = load_test_data(DATASET_DIR)
    X_sample, y_sample = sample_data(X_test, y_test, SAMPLE_SIZE)

    # Combine X_sample and y_sample for the dashboard data
    analysis_df = pd.concat([X_sample, y_sample], axis=1)

    # Save the sample for the dashboard
    output_csv_path = PROJECT_ROOT / "data_for_analysis.csv"
    analysis_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved sample data for analysis to {output_csv_path.resolve()}")

    print("--- DEBUG: X_sample head ---")
    print(X_sample.head())
    print("--- END DEBUG ---")

    try:
        print("--- DEBUG: Accessing underlying sklearn model ---")
        pipeline = model._model_impl.sklearn_model
        print(f"--- DEBUG: Pipeline object: {pipeline} ---")
        
        preprocessor = pipeline.steps[0][1]
        classifier = pipeline.steps[1][1]
        
        print(f"--- DEBUG: Preprocessor: {preprocessor} ---")
        print(f"--- DEBUG: Classifier: {classifier} ---")

        feature_names = preprocessor.get_feature_names_out(input_features=X_sample.columns)
        print(f"--- DEBUG: Extracted {len(feature_names)} feature names ---")
        print(feature_names)
        
        X_sample_processed = pd.DataFrame(
            preprocessor.transform(X_sample), 
            index=X_sample.index, 
            columns=feature_names
        )
        print("--- DEBUG: X_sample_processed head ---")
        print(X_sample_processed.head())
        print("--- END DEBUG ---")

    except Exception as e:
        logger.error(f"Error extracting model components or preprocessing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Initializing SHAP TreeExplainer and calculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample_processed)
        
        # If shap_values is a 3D array, convert it to a list of 2D arrays
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

        # Robustness Check: Ensure SHAP values are in the expected list format
        if not isinstance(shap_values, list) or len(shap_values) != len(classifier.classes_):
             raise TypeError(f"Expected shap_values to be a list of {len(classifier.classes_)} arrays, but got {type(shap_values)} with length {len(shap_values) if isinstance(shap_values, list) else 'N/A'} instead.")
        
        logger.info("SHAP values calculated successfully.")
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}", exc_info=True)
        sys.exit(1)

    top_features = get_top_features_by_shap(shap_values, feature_names, TOP_N_FEATURES)
    class_names = [f"Class_{c}" for c in classifier.classes_]
    
    generate_summary_plots(shap_values, X_sample_processed, class_names, OUTPUT_DIR)
    generate_dependency_plots(shap_values, X_sample_processed, top_features, class_names, OUTPUT_DIR)
    save_global_shap_importance(shap_values, feature_names, class_names, MODEL_DIR)

    logger.info("SHAP analysis complete.")

if __name__ == "__main__":
    main()