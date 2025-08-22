import os
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# -------------------
# Existing helper functions (unchanged)
# -------------------


def ensure_mlflow_from_env(cache_dir: str):
    """
    Use MLFLOW_TRACKING_URI if present (set by run_pipeline.configure_mlflow),
    otherwise set a file:// URI under cache_dir (Path.as_uri() for Windows).
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        try:
            mlflow.set_tracking_uri(env_uri)
        except Exception:
            pass
        return mlflow.get_tracking_uri()
        
    # fallback to file under cache_dir
    try:
        mlruns_dir = os.path.join(os.path.abspath(cache_dir), "mlruns")
        mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
    except Exception:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(cache_dir)}")
    return mlflow.get_tracking_uri()

class FeatureEngineeringPipeline:
    def __init__(self, n_select: int = 50, cor_val: float = 0.7, target_col: str = 'TARGET', cache_dir=None):
        self.n_select = n_select
        self.n_create = max(2, int(np.sqrt(n_select)))
        self.cor_val = cor_val
        self.target_col = target_col
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _validate_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        if not set(df[self.target_col].unique()).issubset({0, 1}):
            raise ValueError("Target column must contain binary values (0 and 1)")
        df = df.copy()
        for col in df.columns:
            if col != self.target_col and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return df

    def run(self, df):
        df = self._validate_input(df)
        # In this refactored version, FeatureEngineeringPipeline no longer performs
        # feature selection, engineering, or intercorrelation dropping globally.
        # It simply validates the input and returns the dataframe.
        return df, pd.DataFrame(index=df.index), df # Return df_select, df_feng, df_combined

# -------------------
# Config dataclass + arg parsing
# -------------------

@dataclass
class Config:
    input_parquet: str = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\df\df_global.parquet"
    output_parquet: str = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\df\df_final.parquet"
    n_select: int = 50
    cor_val: float = 0.7
    completeness: int = 85
    impute: str = "median"
    percent_outliers: int = 1
    cache_dir: Optional[str] = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\cache"
    target_col: str = "TARGET"
    verbose: bool = True
    variance_threshold: float = 0.01
    # scaling options
    scale: bool = False
    scaler: str = "standard"  # allowed: "standard","robust","minmax"

    @staticmethod
    def from_args_and_env():
        parser = argparse.ArgumentParser(description="Process df_global and run feature engineering")
        parser.add_argument("--input", dest="input_parquet", help="Input df_global parquet path")
        parser.add_argument("--output", dest="output_parquet", help="Output processed parquet path")
        parser.add_argument("--n_select", type=int)
        parser.add_argument("--cor_val", type=float)
        parser.add_argument("--completeness", type=int)
        parser.add_argument("--impute", type=str, choices=["median", "mean", "zero"])
        parser.add_argument("--percent_outliers", type=int)
        parser.add_argument("--cache-dir", dest="cache_dir")
        parser.add_argument("--target", dest="target_col")
        parser.add_argument("--variance-threshold", dest="variance_threshold", type=float, help="Drop numeric columns with variance <= this value")
        parser.add_argument("--config-json", dest="config_json", help="Optional JSON config file to load")
        args = parser.parse_args()

        cfg = Config()

        # load config json first (lowest priority)
        if getattr(args, "config_json", None):
            try:
                with open(args.config_json, "r", encoding="utf-8") as f:
                    j = json.load(f)
                for k, v in j.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception as e:
                print(f"Warning: could not load config json {args.config_json}: {e}")

        # overlay CLI args (higher priority)
        for field_name in ("input_parquet", "output_parquet", "n_select", "cor_val", "completeness",
                           "impute", "percent_outliers", "cache_dir", "target_col", "variance_threshold"):
            val = getattr(args, field_name, None)
            if val is not None:
                setattr(cfg, field_name, val)

        # overlay environment variables (highest priority). Prefix: CS_
        env_map = {
            "CS_INPUT": "input_parquet",
            "CS_OUTPUT": "output_parquet",
            "CS_N_SELECT": "n_select",
            "CS_COR_VAL": "cor_val",
            "CS_COMPLETENESS": "completeness",
            "CS_IMPUTE": "impute",
            "CS_PERCENT_OUTLIERS": "percent_outliers",
            "CS_CACHE_DIR": "cache_dir",
            "CS_TARGET": "target_col",
            "CS_VARIANCE_THRESHOLD": "variance_threshold"
        }
        for env_key, cfg_key in env_map.items():
            if env_key in os.environ:
                val = os.environ[env_key]
                # cast numeric types where needed
                if cfg_key in ("n_select", "completeness", "percent_outliers"):
                    try:
                        val = int(val)
                    except Exception:
                        pass
                if cfg_key == "cor_val" or cfg_key == "variance_threshold":
                    try:
                        val = float(val)
                    except Exception:
                        pass
                setattr(cfg, cfg_key, val)

        return cfg

# -------------------
# Updated main that accepts Config
# -------------------

def validate_process_config(cfg):
    """Validate process_df_global.Config instance; raise ValueError on fatal problems."""
    errors = []
    # input file must exist
    if not cfg.input_parquet or not os.path.exists(cfg.input_parquet):
        errors.append(f"input_parquet missing or not found: {cfg.input_parquet}")
        
    # output dir must be creatable
    out_dir = os.path.dirname(cfg.output_parquet) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"output_parquet dir not creatable: {out_dir} -> {e}")
        
    # numerical ranges
    if not (0 <= cfg.completeness <= 100):
        errors.append(f"completeness must be 0-100, got: {cfg.completeness}")
    if cfg.n_select is None or cfg.n_select <= 0:
        errors.append(f"n_select must be positive integer, got: {cfg.n_select}")
    if not (0.0 <= cfg.cor_val <= 1.0):
        errors.append(f"cor_val must be between 0 and 1, got: {cfg.cor_val}")
    if cfg.impute not in ("median","mean","mode","zero","ffill","bfill","drop","raise"):
        errors.append(f"impute policy unknown: {cfg.impute}")
    if cfg.variance_threshold is None or cfg.variance_threshold < 0:
        errors.append(f"variance_threshold must be >= 0, got: {cfg.variance_threshold}")
        
    if errors:
        msg = "process_df_global config validation failed:\n  " + "\n  ".join(errors)
        print(f"[process_df_global][VALIDATION] {msg}")
        raise ValueError(msg)
        
    print(f"[process_df_global][VALIDATION] OK: input={cfg.input_parquet}, output={cfg.output_parquet}, impute={cfg.impute}")

def main_cfg(cfg: Config):
    validate_process_config(cfg)
    print("Configuration:")
    print(json.dumps({
        "input_parquet": cfg.input_parquet,
        "output_parquet": cfg.output_parquet,
        "n_select": cfg.n_select,
        "cor_val": cfg.cor_val,
        "completeness": cfg.completeness,
        "impute": cfg.impute,
        "percent_outliers": cfg.percent_outliers,
        "cache_dir": cfg.cache_dir,
        "target_col": cfg.target_col,
        "variance_threshold": cfg.variance_threshold,
        "scale": cfg.scale,
        "scaler": cfg.scaler
    }, indent=2))
    
    print("Loading df_global from:", cfg.input_parquet)
    df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
    print(f"[process_df_global] Loaded df: rows={len(df)}, cols={len(df.columns)}")
    df_cleaned = clean_and_impute_data(df, target_col=cfg.target_col, completeness=cfg.completeness, impute=cfg.impute, verbose=cfg.verbose, variance_threshold=cfg.variance_threshold)
    print(f"[process_df_global] After clean_and_impute_data: rows={len(df_cleaned)}, cols={len(df_cleaned.columns)}")
    df_cleaned = remove_percent_outliers_2sides(df_cleaned, percent=cfg.percent_outliers)
    print(f"[process_df_global] After remove_percent_outliers_2sides: rows={len(df_cleaned)}, cols={len(df_cleaned.columns)}")
    
    # optional scaling BEFORE feature engineering
    if cfg.scale:
        print(f"[process_df_global] Scaling enabled: scaler='{cfg.scaler}'")
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        # exclude target from scaling
        if cfg.target_col in num_cols:
            num_cols.remove(cfg.target_col)
        if num_cols:
            print(f"[process_df_global] Numeric cols to scale: {len(num_cols)} (showing up to 10): {num_cols[:10]}")
            # compute pre-scaling diagnostics
            pre_stats = df_cleaned[num_cols].describe().to_dict()
            dbg_pre = os.path.join(cfg.cache_dir or ".", "scaling_pre_stats.json")
            try:
                with open(dbg_pre, "w", encoding="utf8") as _f:
                    json.dump(pre_stats, _f, indent=2, default=int)
                print(f"[process_df_global] Saved pre-scaling stats -> {dbg_pre}")
            except Exception as e:
                print(f"[process_df_global] Warning: could not save pre-scaling stats: {e}")

            if cfg.scaler == "standard":
                scaler_obj = StandardScaler()
            elif cfg.scaler == "robust":
                scaler_obj = RobustScaler()
            elif cfg.scaler == "minmax":
                scaler_obj = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler '{cfg.scaler}'. Choose from 'standard','robust','minmax'.")

            # fit_transform on numeric columns; keep index/columns intact
            try:
                scaled_vals = scaler_obj.fit_transform(df_cleaned[num_cols].astype(float))
                print(f"[DEBUG] Scaler parameters (mean/scale) learned from entire dataset: mean={getattr(scaler_obj, 'mean_', 'N/A')}, scale={getattr(scaler_obj, 'scale_', 'N/A')}")
                df_cleaned.loc[:, num_cols] = scaled_vals
                # save scaler for reproducibility
                scaler_fp = os.path.join(cfg.cache_dir or ".", "scaler.joblib")
                try:
                    joblib.dump(scaler_obj, scaler_fp)
                    print(f"[process_df_global] Saved scaler -> {scaler_fp}")
                except Exception as e:
                    print(f"[process_df_global] Warning: could not save scaler: {e}")
                # post-scaling diagnostics
                post_stats = pd.DataFrame(df_cleaned[num_cols]).describe().to_dict()
                dbg_post = os.path.join(cfg.cache_dir or ".", "scaling_post_stats.json")
                try:
                    with open(dbg_post, "w", encoding="utf8") as _f:
                        json.dump(post_stats, _f, indent=2, default=int)
                    print(f"[process_df_global] Saved post-scaling stats -> {dbg_post}")
                except Exception as e:
                    print(f"[process_df_global] Warning: could not save post-scaling stats: {e}")
            except Exception as e:
                print(f"[process_df_global] ERROR during scaling: {e}")
                raise
        else:
            print("[process_df_global] No numeric columns to scale.")
    else:
        print("[process_df_global] Scaling disabled (cfg.scale=False).")
    
    print("Running feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline(n_select=cfg.n_select, cor_val=cfg.cor_val, target_col=cfg.target_col, cache_dir=cfg.cache_dir)
    _, _, df_final = pipeline.run(df_cleaned.copy())

    # --- NEW: enforce completeness, variance threshold, imputation + handle rows w/o target ---
    print(f"[process_df_global] Starting post-processing: completeness={cfg.completeness}%, variance_threshold={cfg.variance_threshold}, impute={cfg.impute}")

    # 1) Drop columns below completeness threshold (percentage of non-null values)
    total_rows = len(df_final)
    if total_rows == 0:
        raise ValueError("[process_df_global] df_final is empty after feature pipeline.")
        
    col_completeness = (1 - df_final.isnull().sum() / total_rows) * 100
    cols_to_keep = col_completeness[col_completeness >= float(cfg.completeness)].index.tolist()
    cols_dropped_for_completeness = [c for c in df_final.columns if c not in cols_to_keep]
    if cols_dropped_for_completeness:
        print(f"[process_df_global] Dropping {len(cols_dropped_for_completeness)} cols for low completeness: {cols_dropped_for_completeness[:10]}")
    df_final = df_final[cols_to_keep].copy()

    # 2) Drop numeric columns whose variance <= variance_threshold
    # NOTE: variance_threshold is expected to be applied once (earlier in clean_and_impute_data
    # or inside FeatureEngineeringPipeline). Do not apply it twice here.
    # If you want variance filtering here instead, remove it from the earlier step.

    # 3) Impute missing values USING ALL ROWS (including those with missing target) so we can fill feature NAs
    def _impute_df(df, strategy):
        df = df.copy()
        if strategy in ("median", "mean"):
            num_cols = df.select_dtypes(include=[np.number]).columns
            if strategy == "median":
                fill_vals = df[num_cols].median(numeric_only=True)
            else:
                fill_vals = df[num_cols].mean(numeric_only=True)
            df[num_cols] = df[num_cols].fillna(fill_vals)
            # categorical: fill with mode
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                mode_vals = df[c].mode(dropna=True)
                if not mode_vals.empty:
                    df[c] = df[c].fillna(mode_vals.iloc[0])
        elif strategy == "mode":
            for c in df.columns:
                mode_vals = df[c].mode(dropna=True)
                if not mode_vals.empty:
                    df[c] = df[c].fillna(mode_vals.iloc[0])
        elif strategy == "zero":
            df = df.fillna(0)
        elif strategy in ("ffill", "bfill"):
            df = df.fillna(method=strategy)
        elif strategy == "drop":
            # keep rows (we still want to use rows without target to impute other rows) -> here drop columns with too many NAs already handled above
            df = df.dropna(axis=0, how="any")
        else:
            # default: median
            num_cols = df.select_dtypes(include=[np.number]).columns
            fill_vals = df[num_cols].median(numeric_only=True)
            df[num_cols] = df[num_cols].fillna(fill_vals)
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                mode_vals = df[c].mode(dropna=True)
                if not mode_vals.empty:
                    df[c] = df[c].fillna(mode_vals.iloc[0])
        return df

    print(f"[process_df_global] Imputing missing feature values using policy: '{cfg.impute}' (rows WITHOUT target are used for imputation)")
    df_imputed = _impute_df(df_final, cfg.impute)

    # 4) After imputation, drop rows that do not have a valid target (we kept them only for imputation)
    if cfg.target_col in df_imputed.columns:
        y_raw = df_imputed[cfg.target_col]
        y_num = pd.to_numeric(y_raw, errors="coerce")
        n_missing_target = int(y_num.isna().sum())
        if n_missing_target > 0:
            print(f"[process_df_global] Dropping {n_missing_target} rows with missing target '{cfg.target_col}' (they were kept for imputation).")
            df_imputed = df_imputed.loc[~y_num.isna()].copy()
        # final safe cast to int
        df_imputed[cfg.target_col] = pd.to_numeric(df_imputed[cfg.target_col], errors="coerce").fillna(0).astype(int)
    else:
        print(f"[process_df_global] Warning: target_col '{cfg.target_col}' not present after pipeline/imputation.")

    # replace df_final with the imputed-and-cleaned dataframe
    df_final = df_imputed
    n_cols = df_final.shape[1]
    n_rows = df_final.shape[0]
    print(f"[process_df_global] Final dataframe shape: rows={n_rows}, cols={n_cols}")

    # --- ALWAYS persist the processed dataframe to output_parquet before MLflow logging ---
    os.makedirs(os.path.dirname(cfg.output_parquet) or ".", exist_ok=True)
    df_final.to_parquet(cfg.output_parquet, engine="pyarrow")
    print(f"[process_df_global] Wrote output_parquet: {cfg.output_parquet}")

    # -- MLflow logging: dataframe metadata + config --
    try:
        # ensure mlruns saved in external cache_dir (use Path.as_uri to produce valid file:// URI on Windows)
        os.makedirs(cfg.cache_dir or ".", exist_ok=True)
        try:
            mlruns_dir = os.path.join(os.path.abspath(cfg.cache_dir or "."), "mlruns")
            mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
        except Exception:
            pass
        mlflow.set_experiment("credit_scoring_process_df")
        with mlflow.start_run(run_name=f"process_df_n{cfg.n_select}_c{cfg.cor_val}"):
            # log basic params
            mlflow.log_param("input_parquet", cfg.input_parquet)
            mlflow.log_param("output_parquet", cfg.output_parquet)
            mlflow.log_param("n_select", cfg.n_select)
            mlflow.log_param("cor_val", cfg.cor_val)
            mlflow.log_param("completeness", cfg.completeness)
            mlflow.log_param("impute", cfg.impute)
            mlflow.log_param("percent_outliers", cfg.percent_outliers)
            mlflow.log_param("variance_threshold", cfg.variance_threshold)
            mlflow.log_param("target_col", cfg.target_col)

            # dataframe metadata
            n_rows, n_cols = df_final.shape
            mlflow.log_metric("df_n_rows", int(n_rows))
            mlflow.log_metric("df_n_cols", int(n_cols))

            # save a small sample and the parquet as artifacts
            os.makedirs(os.path.dirname(cfg.output_parquet), exist_ok=True)
            df_final.to_parquet(cfg.output_parquet, engine="pyarrow")
            sample_path = os.path.join(cfg.cache_dir or ".", "df_final_sample.csv")
            df_final.head(200).to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path, artifact_path="dataframe_samples")
            # log the full parquet as artifact (stored in external output location)
            mlflow.log_artifact(cfg.output_parquet, artifact_path="dataframes")
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")

# backward-compatible CLI entrypoint
def main(input_parquet, output_parquet, n_select=50, cor_val=0.7):
    cfg = Config(input_parquet=input_parquet, output_parquet=output_parquet,
                 n_select=n_select, cor_val=cor_val)
    main_cfg(cfg)

if __name__ == "__main__":
    cfg = Config.from_args_and_env()
    main_cfg(cfg)