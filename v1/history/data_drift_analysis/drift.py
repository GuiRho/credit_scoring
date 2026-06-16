"""
Data Drift Analysis Script : Generates a drift report using Evidently.

This script compares the training dataset (reference) with the test dataset 
(current) to detect data drift.

It expects:
1.  The processed training dataset ('train_processed.parquet') in the DATASET_DIR.
2.  The processed test dataset ('test_processed.parquet') in the DATASET_DIR.
"""
import logging
from pathlib import Path
import sys
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

# --- CONFIGURATION ---

# 1. Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define Paths Relative to Project Root
# Directory containing the processed datasets.
DATASET_DIR = "C:/Users/gui/Documents/OpenClassrooms/Projet 7/cache/processed_s80_c60_robust"
# Directory where the drift report will be saved.
OUTPUT_DIR = PROJECT_ROOT / "data_drift_analysis"
REPORT_FILENAME = "data_drift_report.html"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DRIFT ANALYSIS FUNCTIONS ---

def load_datasets_for_drift(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training (reference) and test (current) datasets.
    
    Args:
        data_dir (str): The absolute path to the directory containing the data.

    Returns:
        A tuple containing the reference DataFrame and the current DataFrame.
    """
    try:
        data_path = Path(data_dir)
        train_path = data_path / "train_processed.parquet"
        test_path = data_path / "test_processed.parquet"

        logger.info(f"Loading reference data from: {train_path}")
        if not train_path.is_file():
            raise FileNotFoundError(f"Reference data 'train_processed.parquet' not found in '{data_dir}'")
        reference_df = pd.read_parquet(train_path)
        logger.info(f"Reference data shape: {reference_df.shape}")

        logger.info(f"Loading current data from: {test_path}")
        if not test_path.is_file():
            raise FileNotFoundError(f"Current data 'test_processed.parquet' not found in '{data_dir}'")
        current_df = pd.read_parquet(test_path)
        logger.info(f"Current data shape: {current_df.shape}")
        
        return reference_df, current_df

    except Exception as e:
        logger.error(f"FATAL: Could not load datasets for drift analysis. Error: {e}", exc_info=True)
        sys.exit(1)

def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: Path):
    """
    Creates and saves a data drift report using Evidently.
    
    Args:
        reference_df (pd.DataFrame): The baseline dataset (e.g., training data).
        current_df (pd.DataFrame): The new dataset to compare (e.g., test data).
        output_path (Path): The full path to save the HTML report file.
    """
    logger.info("Generating data drift report...")
    
    try:
        # The DataDriftPreset provides a comprehensive overview of drift.
        drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        # Run the comparison. The 'TARGET' column will be automatically handled.
        drift_report.run(reference_data=reference_df, current_data=current_df)
        
        # Save the report to the specified file.
        drift_report.save_html(str(output_path))
        
        logger.info(f"Data drift report successfully saved to: {output_path.resolve()}")

    except Exception as e:
        logger.error(f"Failed to generate Evidently report. Error: {e}", exc_info=True)
        sys.exit(1)

def main():
    """Main function to run the data drift analysis."""
    logger.info("Starting Data Drift Analysis...")
    
    # Ensure the output directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / REPORT_FILENAME
    
    # 1. Load the datasets
    reference_data, current_data = load_datasets_for_drift(DATASET_DIR)
    
    # 2. Generate and save the report
    generate_drift_report(reference_data, current_data, report_path)
    
    logger.info("Data Drift Analysis complete.")

if __name__ == "__main__":
    main()