import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

# -------------------
# Helper functions
# -------------------

def clean_and_impute_data(df, target_col='TARGET', completeness=85, impute='median', verbose=True, variance_threshold: float = 0.0):
    df_processed = df.copy()
    initial_cols = df_processed.shape[1]
    initial_rows = df_processed.shape[0]
    col_completeness = (1 - df_processed.isnull().sum() / len(df_processed)) * 100
    cols_to_drop_completeness = col_completeness[col_completeness < completeness].index.tolist()
    if cols_to_drop_completeness:
        df_processed.drop(columns=cols_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(cols_to_drop_completeness)} columns due to completeness < {completeness}%")
            
    row_completeness = (1 - df_processed.isnull().sum(axis=1) / df_processed.shape[1]) * 100
    rows_to_drop_completeness = df_processed[row_completeness < completeness*0.5].index.tolist()
    if rows_to_drop_completeness:
        df_processed.drop(index=rows_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(rows_to_drop_completeness)} rows due to completeness < {completeness*0.5}%")
            
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if impute == 'median':
        impute_value = df_processed[numerical_cols].median()
    elif impute == 'mean':
        impute_value = df_processed[numerical_cols].mean()
    elif impute == 'zero':
        impute_value = 0
    else:
        raise ValueError(f"Unknown impute method: {impute}")
        
    df_processed[numerical_cols] = df_processed[numerical_cols].fillna(impute_value)
    
    if variance_threshold is not None and variance_threshold > 0:
        numerical_cols_after = df_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols_after:
            variances = df_processed[numerical_cols_after].var()
            cols_to_drop_variance = variances[variances <= variance_threshold].index.tolist()
            if cols_to_drop_variance:
                df_processed.drop(columns=cols_to_drop_variance, inplace=True)
                if verbose:
                    print(f"Dropped {len(cols_to_drop_variance)} numerical columns with variance <= {variance_threshold}")
                    
    if target_col in df_processed.columns:
        df_processed.dropna(subset=[target_col], inplace=True)
        df_processed[target_col] = df_processed[target_col].astype(int)
        
    if verbose:
        print(f"Original shape: ({initial_rows}, {initial_cols})")
        print(f"Processed shape: {df_processed.shape}")
        
    return df_processed

def remove_percent_outliers_2sides(df, percent):
    df_cleaned = df.copy()
    for feat in df_cleaned.columns:
        df_cleaned[feat] = pd.to_numeric(df_cleaned[feat], errors='coerce')
    df_cleaned = df_cleaned.dropna()
    
    if df_cleaned.empty:
        return df_cleaned
        
    all_outlier_indices = []
    lower_quantile = percent / 100
    upper_quantile = (100 - percent) / 100
    
    for feat in df_cleaned.select_dtypes(include=[np.number]).columns:
        lower_val = df_cleaned[feat].quantile(lower_quantile)
        upper_val = df_cleaned[feat].quantile(upper_quantile)
        outlier_rows = df_cleaned[(df_cleaned[feat] < lower_val) | (df_cleaned[feat] > upper_val)]
        all_outlier_indices.extend(outlier_rows.index.tolist())
        
    all_outlier_indices = list(set(all_outlier_indices))
    print(f"Number of total outliers = {len(all_outlier_indices)}")
    df_cleaned = df_cleaned.drop(index=all_outlier_indices)
    return df_cleaned

# -------------------
# Config and Main
# -------------------

@dataclass
class Config:
    input_parquet: str
    output_parquet: str
    completeness: int = 85
    impute: str = "median"
    percent_outliers: int = 1
    target_col: str = "TARGET"
    verbose: bool = True
    variance_threshold: float = 0.01

def main(cfg: Config):
    print("Loading data from:", cfg.input_parquet)
    df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
    
    df_cleaned = clean_and_impute_data(
        df, 
        target_col=cfg.target_col, 
        completeness=cfg.completeness, 
        impute=cfg.impute, 
        verbose=cfg.verbose, 
        variance_threshold=cfg.variance_threshold
    )
    
    df_cleaned = remove_percent_outliers_2sides(df_cleaned, percent=cfg.percent_outliers)
    
    print("Saving cleaned data to:", cfg.output_parquet)
    df_cleaned.to_parquet(cfg.output_parquet, engine="pyarrow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data: clean, impute, and remove outliers.")
    parser.add_argument("--input", required=True, dest="input_parquet", help="Input parquet path")
    parser.add_argument("--output", required=True, dest="output_parquet", help="Output processed parquet path")
    parser.add_argument("--completeness", type=int, default=85)
    parser.add_argument("--impute", type=str, default="median", choices=["median", "mean", "zero"])
    parser.add_argument("--percent_outliers", type=int, default=1)
    parser.add_argument("--target", dest="target_col", default="TARGET")
    parser.add_argument("--variance-threshold", dest="variance_threshold", type=float, default=0.01)
    
    args = parser.parse_args()
    
    config = Config(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        completeness=args.completeness,
        impute=args.impute,
        percent_outliers=args.percent_outliers,
        target_col=args.target_col,
        variance_threshold=args.variance_threshold
    )
    
    main(config)
