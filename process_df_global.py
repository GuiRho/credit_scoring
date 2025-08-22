import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow

# --------------------
# Existing helper functions (unchanged)
# --------------------

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
    # apply low-variance filter on numerical columns (after imputation)
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

class FeatureEngineeringPipeline:
    def __init__(self, n_select: int = 50, cor_val: float = 0.7, target_col: str = 'TARGET', cache_dir=None):
        self.n_select = n_select
        self.n_create = max(2, int(np.sqrt(n_select)))
        self.cor_val = cor_val
        self.target_col = target_col
        self.importance_df = None
        self.feng_importance_df = None
        self.combined_importance_df = None
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

    def _calcul_feature_importance(self, df, cache_key):
        if df.empty or df.shape[1] <= 1:
            return pd.DataFrame(columns=['feature', 'metric1_spearman', 'metric2_mdi', 'metric3_product'])
        cache_file = os.path.join(self.cache_dir, f"importance_{cache_key}.pkl")
        if os.path.exists(cache_file):
            cached_df = joblib.load(cache_file)
            current_features = set(df.drop(columns=[self.target_col], errors='ignore').columns)
            cached_features = set(cached_df['feature'])
            if current_features == cached_features:
                return cached_df
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        if X.empty:
            return pd.DataFrame(columns=['feature', 'metric1_spearman', 'metric2_mdi', 'metric3_product'])
        spearman_corr = np.abs(X.corrwith(y, method='spearman').fillna(0))
        rfc = RandomForestClassifier(n_estimators=50, random_state=42)
        rfc.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'metric1_spearman': spearman_corr,
            'metric2_mdi': rfc.feature_importances_,
            'metric3_product': spearman_corr * rfc.feature_importances_
        }).sort_values(by='metric3_product', ascending=False).reset_index(drop=True)
        joblib.dump(importance_df, cache_file)
        return importance_df

    def _top_feature_selection(self):
        if self.importance_df is None or self.importance_df.empty:
            return [], []
        top_metric1 = self.importance_df.nlargest(self.n_select, 'metric1_spearman')['feature'].tolist()
        top_metric2 = self.importance_df.nlargest(self.n_select, 'metric2_mdi')['feature'].tolist()
        top_metric3 = self.importance_df.nlargest(self.n_select, 'metric3_product')['feature'].tolist()
        list_select = sorted(list(set(top_metric1 + top_metric2 + top_metric3)))
        top_create_metric1 = self.importance_df.nlargest(self.n_create, 'metric1_spearman')['feature'].tolist()
        top_create_metric2 = self.importance_df.nlargest(self.n_create, 'metric2_mdi')['feature'].tolist()
        top_create_metric3 = self.importance_df.nlargest(self.n_create, 'metric3_product')['feature'].tolist()
        list_create = sorted(list(set(top_create_metric1 + top_create_metric2 + top_create_metric3)))
        return list_select, list_create

    def _create_new_features(self, df, feature_list, epsilon=1e-6):
        if not feature_list:
            return pd.DataFrame(index=df.index)
        new_features_list = []
        for feature in feature_list:
            if feature in df.columns:
                feature_values_abs = df[feature].abs()
                new_features_list.append(pd.Series(np.sqrt(feature_values_abs), name=f'{feature}_pow0_5', index=df.index))
                new_features_list.append(pd.Series(df[feature] ** 2, name=f'{feature}_pow2', index=df.index))
                new_features_list.append(pd.Series(np.log(feature_values_abs + epsilon), name=f'{feature}_log', index=df.index))
        from itertools import combinations
        for f1, f2 in combinations(feature_list, 2):
            if f1 in df.columns and f2 in df.columns:
                new_features_list.append(pd.Series(df[f1] + df[f2], name=f'{f1}_plus_{f2}', index=df.index))
                new_features_list.append(pd.Series(df[f1] * df[f2], name=f'{f1}_times_{f2}', index=df.index))
        if not new_features_list:
            return pd.DataFrame(index=df.index)
        return pd.concat(new_features_list, axis=1)

    def _drop_intercorrelated(self, df, importance_df):
        if df.empty or importance_df.empty or len(df.columns) < 2:
            return df
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return df
        corr_matrix = numeric_df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            correlated_features = upper.index[upper[col] > self.cor_val].tolist()
            if correlated_features:
                all_correlated = [col] + correlated_features
                importance_subset = importance_df[importance_df['feature'].isin(all_correlated)]
                importance_subset = importance_subset[importance_subset['feature'].isin(df.columns)]
                if not importance_subset.empty:
                    importance_subset = importance_subset.copy()
                    importance_subset['metric3_product'] = pd.to_numeric(importance_subset['metric3_product'], errors='coerce').fillna(0)
                    feature_to_keep = importance_subset.loc[importance_subset['metric3_product'].idxmax()]['feature']
                    to_drop.update(f for f in all_correlated if f != feature_to_keep and f in df.columns)
        return df.drop(columns=list(to_drop), errors='ignore')

    def run(self, df):
        df = self._validate_input(df)
        run_hash = f"n{self.n_select}_c{str(self.cor_val).replace('.', '')}"
        self.importance_df = self._calcul_feature_importance(df, cache_key=run_hash)
        n_select_list, n_create_list = self._top_feature_selection()
        existing_n_select_list = [col for col in n_select_list if col in df.columns]
        df_select = self._drop_intercorrelated(df[existing_n_select_list], self.importance_df)
        df_select = df_select.join(df[[self.target_col]])
        existing_n_create_list = [col for col in n_create_list if col in df.columns]
        df_feng_initial = self._create_new_features(df, existing_n_create_list)
        if not df_feng_initial.empty:
            self.feng_importance_df = self._calcul_feature_importance(df_feng_initial.join(df[[self.target_col]]), cache_key=f"{run_hash}_feng")
            df_feng = self._drop_intercorrelated(df_feng_initial, self.feng_importance_df)
        else:
            df_feng = pd.DataFrame(index=df.index)
        df_feng = df_feng.join(df[[self.target_col]])
        cols_to_drop_select = [self.target_col] if self.target_col in df_select.columns else []
        cols_to_drop_feng = [self.target_col] if self.target_col in df_feng.columns else []
        df_combined_initial = pd.concat([
            df_select.drop(columns=cols_to_drop_select, errors='ignore'),
            df_feng.drop(columns=cols_to_drop_feng, errors='ignore')
        ], axis=1)
        if not df_combined_initial.empty:
            self.combined_importance_df = self._calcul_feature_importance(df_combined_initial.join(df[[self.target_col]]), cache_key=f"{run_hash}_combined")
            df_combined = self._drop_intercorrelated(df_combined_initial, self.combined_importance_df)
        else:
            df_combined = pd.DataFrame(index=df.index)
        df_combined = df_combined.join(df[[self.target_col]])
        return df_select, df_feng, df_combined

# --------------------
# Config dataclass + arg parsing
# --------------------

@dataclass
class Config:
    input_parquet: str = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\df\df_global.parquet"
    output_parquet: str = r"./df_final.parquet"
    n_select: int = 50
    cor_val: float = 0.7
    completeness: int = 85
    impute: str = "median"
    percent_outliers: int = 1
    cache_dir: Optional[str] = None
    target_col: str = "TARGET"
    verbose: bool = True
    variance_threshold: float = 0.01

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

# --------------------
# Updated main that accepts Config
# --------------------

def main_cfg(cfg: Config):
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
        "variance_threshold": cfg.variance_threshold
    }, indent=2))
    print("Loading df_global from:", cfg.input_parquet)
    df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
    df_cleaned = clean_and_impute_data(df, target_col=cfg.target_col, completeness=cfg.completeness, impute=cfg.impute, verbose=cfg.verbose, variance_threshold=cfg.variance_threshold)
    df_cleaned = remove_percent_outliers_2sides(df_cleaned, percent=cfg.percent_outliers)
    print("Running feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline(n_select=cfg.n_select, cor_val=cfg.cor_val, target_col=cfg.target_col, cache_dir=cfg.cache_dir)
    _, _, df_final = pipeline.run(df_cleaned.copy())

    # ensure output dir exists
    os.makedirs(os.path.dirname(cfg.output_parquet) or ".", exist_ok=True)

    # reset index for reproducibility and ensure target is present and numeric
    df_final = df_final.reset_index(drop=True)
    if cfg.target_col in df_final.columns:
        df_final[cfg.target_col] = pd.to_numeric(df_final[cfg.target_col], errors='coerce').fillna(0).astype(int)

    df_final.to_parquet(cfg.output_parquet, engine='pyarrow')
    print(f"Saved processed output to {cfg.output_parquet}")

    # -- MLflow logging: dataframe metadata + config --
    try:
        import mlflow
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
            sample_path = os.path.join(cfg.cache_dir or ".", "df_final_sample.csv")
            df_final.head(200).to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path, artifact_path="dataframe_samples")

            # log the full parquet as artifact
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