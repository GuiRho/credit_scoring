import os
import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("shap_analysis.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "CreditScoringRF"
MODEL_ARTIFACT_PATH = "credit_scoring_model"
SAMPLE_SIZE = 100  # Configurable sample size
PLOT_DPI = 300  # High resolution for plots

# --- MLflow Setup ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_production_model_details(registered_model_name, model_artifact_path=MODEL_ARTIFACT_PATH):
    """
    Retrieves the latest production model version and its run ID.
    
    Args:
        registered_model_name (str): Name of the registered model in MLflow.
        model_artifact_path (str): Path to the model artifact in the MLflow run.
    
    Returns:
        tuple: (model_uri, run_id) or (None, None) if an error occurs.
    """
    logger.info(f"Fetching production model details for '{registered_model_name}'...")
    try:
        prod_versions = client.get_latest_versions(
            name=registered_model_name, stages=["Production"]
        )
        if not prod_versions:
            raise ValueError(
                f"No model version found for '{registered_model_name}' in stage 'Production'."
            )
        latest_prod_version = prod_versions[0]
        run_id = latest_prod_version.run_id
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        logger.info(f"Found production model version {latest_prod_version.version} from run_id: {run_id}")
        return model_uri, run_id
    except Exception as e:
        logger.error(f"Error fetching model details: {e}")
        return None, None


def load_data_from_run(run_id):
    """
    Loads data artifacts (X_train, y_train, X_test, y_test) from a specific MLflow run.
    
    Args:
        run_id (str): MLflow run ID containing the data artifacts.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) or (None, None, None, None) if an error occurs.
    """
    logger.info(f"Loading data artifacts from run_id: {run_id}")
    try:
        local_dir = client.download_artifacts(run_id, "processed_data")
        required_files = ["X_train.parquet", "y_train.parquet", "X_test.parquet", "y_test.parquet"]
        
        # Check if all required files exist
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(local_dir, file))]
        if missing_files:
            raise FileNotFoundError(f"Missing artifacts: {', '.join(missing_files)} in 'processed_data' directory.")
        
        X_train = pd.read_parquet(os.path.join(local_dir, "X_train.parquet"))
        y_train = pd.read_parquet(os.path.join(local_dir, "y_train.parquet")).squeeze()
        X_test = pd.read_parquet(os.path.join(local_dir, "X_test.parquet"))
        y_test = pd.read_parquet(os.path.join(local_dir, "y_test.parquet")).squeeze()
        
        # Validate data shapes
        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
            raise ValueError("Mismatch between feature and target dimensions")
            
        logger.info("Successfully loaded X_train, y_train, X_test, and y_test data.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Error loading data artifacts: {e}")
        return None, None, None, None


def setup_shap_environment():
    """Set up SHAP environment with consistent styling."""
    plt.style.use('default')
    shap.initjs()
    # Removed invalid rcParams settings


def validate_model_structure(model):
    """
    Validate the model structure and extract components.
    
    Args:
        model: Loaded MLflow model
        
    Returns:
        tuple: (rf_classifier, preprocessor_pipeline)
    """
    rf_classifier = None
    preprocessor_pipeline = None
    
    if isinstance(model, Pipeline):
        # Extract classifier
        if 'clf' in model.named_steps and isinstance(model.named_steps['clf'], RandomForestClassifier):
            rf_classifier = model.named_steps['clf']
            # Extract preprocessing steps
            preprocessor_steps = [(name, step) for name, step in model.named_steps.items() if name != 'clf']
            if preprocessor_steps:
                preprocessor_pipeline = Pipeline(preprocessor_steps)
                logger.info("Identified preprocessing pipeline steps.")
            else:
                logger.info("No preprocessing steps identified before the classifier in the pipeline.")
        else:
            raise ValueError("The MLflow model is a Pipeline but does not contain a 'clf' step "
                             "which is a RandomForestClassifier.")
    elif isinstance(model, RandomForestClassifier):
        rf_classifier = model
        logger.info("Loaded model is directly a RandomForestClassifier (not a Pipeline).")
    else:
        raise ValueError("The MLflow model is not a Pipeline and not a RandomForestClassifier.")
    
    return rf_classifier, preprocessor_pipeline


def get_feature_names(preprocessor_pipeline, X_train):
    """
    Extract feature names after preprocessing.
    
    Args:
        preprocessor_pipeline: Preprocessing pipeline
        X_train: Training features
        
    Returns:
        list: Feature names after preprocessing
    """
    if preprocessor_pipeline:
        logger.info("Applying preprocessing to infer transformed feature names...")
        try:
            if not hasattr(preprocessor_pipeline, 'feature_names_in_'): 
                preprocessor_pipeline.fit(X_train)
                logger.info("Preprocessor pipeline fitted for feature name inference.")
            
            if isinstance(preprocessor_pipeline.steps[-1][1], ColumnTransformer):
                transformed_X = preprocessor_pipeline.transform(X_train.iloc[:1])
                try:
                    shap_feature_names = preprocessor_pipeline.steps[-1][1].get_feature_names_out(X_train.columns).tolist()
                    logger.info("Successfully inferred transformed feature names using ColumnTransformer's get_feature_names_out().")
                except AttributeError:
                    logger.warning("ColumnTransformer does not have get_feature_names_out with input columns. Falling back.")
                    if hasattr(preprocessor_pipeline, 'get_feature_names_out'):
                         shap_feature_names = preprocessor_pipeline.get_feature_names_out().tolist()
                         logger.info("Successfully inferred transformed feature names using pipeline's get_feature_names_out().")
                    elif transformed_X.shape[1] == X_train.shape[1] and X_train.columns is not None:
                        logger.info("Number of features unchanged after preprocessing. Using original feature names.")
                        shap_feature_names = X_train.columns.tolist()
                    else:
                        logger.warning(f"Transformed data has {transformed_X.shape[1]} features, "
                                      f"original data has {X_train.shape[1]} features. Using generic feature names.")
                        shap_feature_names = [f'feature_{i}' for i in range(transformed_X.shape[1])]
            else:
                if hasattr(preprocessor_pipeline, 'get_feature_names_out'):
                    shap_feature_names = preprocessor_pipeline.get_feature_names_out().tolist()
                    logger.info("Successfully inferred transformed feature names using pipeline's get_feature_names_out().")
                else:
                    transformed_X = preprocessor_pipeline.transform(X_train.iloc[:1])
                    if transformed_X.shape[1] == X_train.shape[1] and X_train.columns is not None:
                        logger.info("Number of features unchanged after preprocessing. Using original feature names.")
                        shap_feature_names = X_train.columns.tolist()
                    else:
                        logger.warning(f"Could not infer transformed feature names using standard methods. "
                                      f"Transformed data has {transformed_X.shape[1]} features, "
                                      f"original data has {X_train.shape[1]} features. Using generic feature names.")
                        shap_feature_names = [f'feature_{i}' for i in range(transformed_X.shape[1])]
        except Exception as e:
            logger.warning(f"Could not infer transformed feature names due to error: {e}. Using original feature names.")
            shap_feature_names = X_train.columns.tolist()
    else:
        logger.info("No preprocessing pipeline identified. Using original feature names.")
        shap_feature_names = X_train.columns.tolist()
    
    logger.info(f"Inferred {len(shap_feature_names)} SHAP feature names.")
    return shap_feature_names


def prepare_shap_data(preprocessor_pipeline, X_sample, shap_feature_names):
    """
    Prepare data for SHAP analysis.
    
    Args:
        preprocessor_pipeline: Preprocessing pipeline (assumed already fitted)
        X_sample: Sample data
        shap_feature_names: Feature names after preprocessing
        
    Returns:
        DataFrame: Processed data for SHAP analysis
    """
    if preprocessor_pipeline:
        X_sample_processed = preprocessor_pipeline.transform(X_sample)
    else:
        X_sample_processed = X_sample.values
    
    if X_sample_processed.shape[1] != len(shap_feature_names):
        logger.error(f"Mismatch: processed data has {X_sample_processed.shape[1]} columns, "
                     f"but {len(shap_feature_names)} feature names were provided. "
                     f"Attempting to proceed but this may cause errors.")
        if X_sample_processed.shape[1] < len(shap_feature_names):
            shap_feature_names = shap_feature_names[:X_sample_processed.shape[1]]
        elif X_sample_processed.shape[1] > len(shap_feature_names):
            shap_feature_names.extend([f'unknown_feature_{i}' for i in range(len(shap_feature_names), X_sample_processed.shape[1])])

    return pd.DataFrame(X_sample_processed, columns=shap_feature_names, index=X_sample.index)


def calculate_feature_importance(rf_classifier, shap_feature_names):
    """
    Calculate and return feature importances.
    
    Args:
        rf_classifier: Random Forest classifier
        shap_feature_names: Feature names
        
    Returns:
        DataFrame: Feature importances sorted by importance
    """
    logger.info("Calculating feature importances (MDI) from the model's classifier...")
    
    if len(rf_classifier.feature_importances_) != len(shap_feature_names):
        logger.error(f"Mismatch: Classifier has {len(rf_classifier.feature_importances_)} importances, "
                     f"but {len(shap_feature_names)} SHAP feature names. This needs to be resolved for accuracy.")
        if len(rf_classifier.feature_importances_) < len(shap_feature_names):
            logger.warning("Truncating SHAP feature names to match classifier importances for importance calculation.")
            shap_feature_names_adjusted = shap_feature_names[:len(rf_classifier.feature_importances_)]
        else:
            logger.warning("Padding SHAP feature names to match classifier importances (this is unusual) for importance calculation.")
            shap_feature_names_adjusted = shap_feature_names + [f'extra_feature_{i}' for i in range(len(shap_feature_names), len(rf_classifier.feature_importances_))]
    else:
        shap_feature_names_adjusted = shap_feature_names

    importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": shap_feature_names_adjusted, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    
    logger.info("Identified feature importances (based on MDI):")
    logger.info(feature_importance_df.head(10).to_string())
    
    return feature_importance_df


def calculate_shap_values(explainer, X_processed, n_classes, class_names):
    """
    Calculate SHAP values with proper handling for different model types and SHAP output formats.
    
    Args:
        explainer: SHAP TreeExplainer
        X_processed: Processed feature data
        n_classes: Number of classes
        class_names: Names of classes
        
    Returns:
        list: SHAP values (2D numpy arrays) for each class
    """
    logger.info("Calculating SHAP values...")
    raw_shap_values = explainer.shap_values(X_processed)

    logger.info(f"DEBUG_RAW_SHAP_OUTPUT: Type of raw_shap_values: {type(raw_shap_values)}")
    if isinstance(raw_shap_values, np.ndarray):
        logger.info(f"DEBUG_RAW_SHAP_OUTPUT: Shape of raw_shap_values: {raw_shap_values.shape}")
    elif isinstance(raw_shap_values, list):
        logger.info(f"DEBUG_RAW_SHAP_OUTPUT: raw_shap_values is a list of {len(raw_shap_values)} arrays.")
        for i, arr in enumerate(raw_shap_values):
            logger.info(f"DEBUG_RAW_SHAP_OUTPUT: List element {i} shape: {arr.shape}")

    final_shap_values_list = []

    # Case 1: Raw SHAP values are a 3D array (samples, features, n_classes)
    if isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3 and raw_shap_values.shape[2] == n_classes:
        logger.info(f"Case: Raw SHAP values are a 3D array (samples, features, {n_classes}). Extracting 2D slice for each class.")
        for i in range(n_classes):
            final_shap_values_list.append(raw_shap_values[:, :, i])
    
    # Case 2: Raw SHAP values are a single 2D array (samples, features) for binary classification
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2 and n_classes == 2:
        logger.info("Case: Raw SHAP values are a single 2D (samples, features) array for binary classification.")
        final_shap_values_list = [(-1) * raw_shap_values, raw_shap_values]
    
    # Case 3: Raw SHAP values are already a list of 2D arrays (samples, features), one per class
    elif isinstance(raw_shap_values, list) and len(raw_shap_values) == n_classes and \
         all(isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == X_processed.shape[1] for s in raw_shap_values):
        logger.info("Case: Raw SHAP values are a list of 2D (samples, features) arrays for multi-class.")
        final_shap_values_list = raw_shap_values
    
    else:
        logger.error(f"UNEXPECTED SHAP values format. Raw Type: {type(raw_shap_values)}, Raw Shape: {raw_shap_values.shape if isinstance(raw_shap_values, np.ndarray) else 'N/A'}. Cannot process correctly.")
        raise ValueError("Unsupported SHAP values format received from explainer.shap_values(). Please check the raw output and adjust `calculate_shap_values` accordingly.")

    # Final validation of the processed SHAP values list
    for i, sv in enumerate(final_shap_values_list):
        if not isinstance(sv, np.ndarray) or sv.ndim != 2 or sv.shape[1] != X_processed.shape[1]:
            logger.error(f"POST-PROCESSING VALIDATION FAILED for class {i}: SHAP values are {type(sv)}, shape {sv.shape if isinstance(sv, np.ndarray) else 'N/A'}. Expected 2D NumPy array with {X_processed.shape[1]} features.")
            raise ValueError(f"SHAP values for class {i} have incorrect dimensions or type after processing. Expected 2D NumPy array with {X_processed.shape[1]} features.")
    
    return final_shap_values_list


def create_output_directory():
    """Create a timestamped directory for outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"shap_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# --- Main SHAP Analysis Script ---
if __name__ == "__main__":
    # Set up environment
    setup_shap_environment()
    output_dir = create_output_directory()
    
    # 0. Setup: Load Model and Data from MLflow
    model_uri, run_id = get_production_model_details(REGISTERED_MODEL_NAME)
    if not model_uri:
        logger.error("Exiting SHAP analysis due to model loading error.")
        sys.exit(1)

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Successfully loaded model pipeline from MLflow.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    X_train, y_train, X_test, y_test = load_data_from_run(run_id)
    if any(x is None for x in [X_train, y_train, X_test, y_test]):
        logger.error("Exiting SHAP analysis due to data loading error.")
        sys.exit(1)

    logger.info(f"Loaded X_test shape: {X_test.shape}")
    logger.info(f"Loaded y_test value counts:\n{y_test.value_counts()}")
    logger.info("-" * 50)

    # 1. Sampling the Data
    logger.info(f"Creating two stratified samples of {SAMPLE_SIZE} rows from the test set...")
    try:
        X_sample1, _, y_sample1, _ = train_test_split(
            X_test, y_test, train_size=SAMPLE_SIZE, stratify=y_test, random_state=42
        )
        X_sample2, _, y_sample2, _ = train_test_split(
            X_test, y_test, train_size=SAMPLE_SIZE, stratify=y_test, random_state=43
        )
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        sys.exit(1)

    logger.info(f"Sample 1 shape: {X_sample1.shape}")
    logger.info(f"Sample 1 target value counts:\n{y_sample1.value_counts()}")
    logger.info(f"Sample 2 shape: {X_sample2.shape}")
    logger.info(f"Sample 2 target value counts:\n{y_sample2.value_counts()}")
    logger.info("-" * 50)

    # 2. Extract model components and feature names
    try:
        rf_classifier, preprocessor_pipeline = validate_model_structure(model)
        if preprocessor_pipeline:
            logger.info("Fitting preprocessor pipeline on X_train for feature name inference and transformation...")
            preprocessor_pipeline.fit(X_train)
            logger.info("Preprocessor pipeline fitted successfully.")
            
        shap_feature_names = get_feature_names(preprocessor_pipeline, X_train)
        logger.info(f"Number of SHAP feature names derived: {len(shap_feature_names)}")
    except Exception as e:
        logger.error(f"Error processing model structure or feature names: {e}")
        sys.exit(1)

    # 3. Preprocess Samples for SHAP Calculation
    logger.info("Applying pipeline preprocessing steps to samples before SHAP calculation...")
    try:
        X_sample1_processed_for_shap = prepare_shap_data(preprocessor_pipeline, X_sample1, shap_feature_names)
        X_sample2_processed_for_shap = prepare_shap_data(preprocessor_pipeline, X_sample2, shap_feature_names)
        logger.info(f"Shape of X_sample1_processed_for_shap: {X_sample1_processed_for_shap.shape}")
        logger.info(f"Shape of X_sample2_processed_for_shap: {X_sample2_processed_for_shap.shape}")
        logger.info(f"Number of columns in X_sample1_processed_for_shap: {X_sample1_processed_for_shap.shape[1]}")
        logger.info(f"Number of SHAP feature names: {len(shap_feature_names)}")
        if X_sample1_processed_for_shap.shape[1] != len(shap_feature_names):
             logger.error("CRITICAL: Mismatch between preprocessed data columns and SHAP feature names length! This will likely cause errors.")
    except Exception as e:
        logger.error(f"Error preprocessing samples: {e}")
        sys.exit(1)

    # 4. Calculate Feature Importances (MDI)
    try:
        feature_importance_df = calculate_feature_importance(rf_classifier, shap_feature_names)
        top_20_features_df = feature_importance_df.head(20)
        top_20_feature_list = top_20_features_df["feature"].tolist()
        top_8_feature_list = top_20_features_df.head(8)["feature"].tolist()
        
        logger.info("Top 20 most important features (based on MDI):")
        logger.info(top_20_features_df.to_string())
        logger.info(f"Top 8 features for dependency plots: {top_8_feature_list}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        sys.exit(1)

    # 5. Initialize SHAP Explainer
    try:
        explainer = shap.TreeExplainer(rf_classifier)
        logger.info("SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SHAP explainer: {e}")
        sys.exit(1)

    # 6. Determine class information
    n_classes = len(np.unique(y_train))
    class_names = [f"Class_{i}" for i in range(n_classes)]
    if hasattr(rf_classifier, 'classes_'):
        class_names = [str(c) for c in rf_classifier.classes_]
    logger.info(f"Detected {n_classes} classes: {class_names}")

    # 7. Calculate SHAP values
    try:
        logger.info("Calculating full SHAP values for Sample 1...")
        shap_values_sample1 = calculate_shap_values(explainer, X_sample1_processed_for_shap, n_classes, class_names)
        
        logger.info("Calculating full SHAP values for Sample 2...")
        shap_values_sample2 = calculate_shap_values(explainer, X_sample2_processed_for_shap, n_classes, class_names)
        
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        sys.exit(1)

    # 8. Generate and Save SHAP Summary Plots (One per class, one per sample)
    logger.info("Generating SHAP Summary Plots (one per sample, per class)...")
    try:
        top_20_feature_indices = [shap_feature_names.index(f) for f in top_20_feature_list]

        # Prepare samples and their names for easier iteration
        samples_for_summary = [
            (shap_values_sample1, X_sample1_processed_for_shap, "Sample 1"),
            (shap_values_sample2, X_sample2_processed_for_shap, "Sample 2")
        ]

        for class_idx in range(n_classes):
            for shap_values_current_sample, X_current_sample_processed, sample_name in samples_for_summary:
                logger.info(f"Generating SHAP summary plot for {class_names[class_idx]} - {sample_name}...")
                
                plt.figure(figsize=(15, 10)) # Adjusted size for a single summary plot
                plt.title(f'SHAP Summary Plot for Top 20 Features - {class_names[class_idx]} ({sample_name})', fontsize=16)

                # Filter SHAP values for the current class and top 20 features
                shap_values_class_top20 = shap_values_current_sample[class_idx][:, top_20_feature_indices]
                X_top20 = X_current_sample_processed[top_20_feature_list] # Get data for top 20 features
                
                shap.summary_plot(shap_values_class_top20, X_top20, show=False)
                
                ax = plt.gca()
                ax.tick_params(axis='y', labelsize=8) # Adjusted to 8 for more space

                plt.tight_layout(rect=[0.25, 0.05, 0.95, 0.95]) # Adjust rect for single plot
                summary_plot_path = os.path.join(output_dir, f"shap_summary_plots_class_{class_names[class_idx]}_{sample_name.replace(' ', '_').lower()}.png")
                plt.savefig(summary_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP Summary Plot for {class_names[class_idx]} - {sample_name} saved to {summary_plot_path}")

        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error generating summary plots: {e}")

    # 9. Generate and Save SHAP Dependency Plots (All 8 features on one plot per class)
    logger.info("Generating SHAP Dependency Plots for the top 8 features for all classes (all on one figure per class)...")
    try:
        n_cols = 4
        n_rows = (len(top_8_feature_list) + n_cols - 1) // n_cols

        for class_idx in range(n_classes):
            logger.info(f"Generating SHAP dependency plots for {class_names[class_idx]}...")
            
            # Create a SINGLE figure for all 8 dependency plots for this class
            # This plt.figure() call is outside the inner feature loop
            plt.figure(figsize=(n_cols * 10, n_rows * 8)) # Increased multipliers for even more space
            plt.suptitle(f'SHAP Dependence Plots for Top 8 Features - {class_names[class_idx]}', fontsize=18)

            for i, feature in enumerate(top_8_feature_list):
                plt.subplot(n_rows, n_cols, i + 1) # Add subplot to the current figure
                
                # --- DEBUGGING PRINTS ---
                logger.info(f"DEBUG: Plotting feature: '{feature}' (index {shap_feature_names.index(feature)}) for class: '{class_names[class_idx]}'")
                
                current_shap_values = shap_values_sample1[class_idx]
                logger.info(f"DEBUG: Shape of shap_values_sample1[{class_idx}]: {current_shap_values.shape}")
                logger.info(f"DEBUG: Type of shap_values_sample1[{class_idx}]: {type(current_shap_values)}")
                logger.info(f"DEBUG: Shape of X_sample1_processed_for_shap: {X_sample1_processed_for_shap.shape}")
                logger.info(f"DEBUG: Type of X_sample1_processed_for_shap: {type(X_sample1_processed_for_shap)}")
                logger.info(f"DEBUG: Total columns in X_sample1_processed_for_shap: {len(X_sample1_processed_for_shap.columns)}")
                if feature not in X_sample1_processed_for_shap.columns:
                    logger.error(f"DEBUG: Feature '{feature}' NOT found in X_sample1_processed_for_shap columns!")
                else:
                    logger.info(f"DEBUG: Feature '{feature}' found in X_sample1_processed_for_shap columns.")
                # --- END DEBUGGING PRINTS ---

                shap.dependence_plot(
                    feature,
                    current_shap_values,
                    X_sample1_processed_for_shap,
                    display_features=X_sample1_processed_for_shap,
                    show=False # Crucial for subplotting
                )
            
            # Save and close the figure ONLY AFTER all subplots for the current class are drawn
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95]) # General padding
            dependency_plot_path = os.path.join(output_dir, f"shap_dependency_plots_class_{class_names[class_idx]}.png")
            plt.savefig(dependency_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP Dependency Plots for {class_names[class_idx]} saved to {dependency_plot_path}")

        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error generating dependency plots: {e}")

    # 10. Save feature importance and other metadata
    try:
        feature_importance_path = os.path.join(output_dir, "feature_importance.csv")
        feature_importance_df.to_csv(feature_importance_path, index=False)
        logger.info(f"Feature importance saved to {feature_importance_path}")
        
        # Save configuration details
        config_details = {
            "sample_size": SAMPLE_SIZE,
            "model_name": REGISTERED_MODEL_NAME,
            "run_id": run_id,
            "n_classes": n_classes,
            "class_names": class_names,
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        config_df = pd.DataFrame.from_dict(config_details, orient='index', columns=['Value'])
        config_path = os.path.join(output_dir, "analysis_config.csv")
        config_df.to_csv(config_path)
        logger.info(f"Analysis configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

    logger.info("Script execution complete.")