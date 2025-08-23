import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow_utils import setup_mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# -------------------
# Feature Engineering Pipeline
# -------------------

class FeatureEngineeringPipeline:
    def __init__(self, n_select: int = 50, cor_val: float = 0.7, target_col: str = 'TARGET', cache_dir: str = None):
        self.n_select = n_select
        self.n_create = max(2, int(np.sqrt(n_select)))
        self.cor_val = cor_val
        self.target_col = target_col
        
        if cache_dir is None:
            raise ValueError("cache_dir must be provided")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Attributes to be learned during fit
        self.importance_df_ = None
        self.feng_importance_df_ = None
        self.combined_importance_df_ = None
        self.n_select_list_ = []
        self.n_create_list_ = []
        self.cols_to_drop_select_ = []
        self.cols_to_drop_feng_ = []
        self.cols_to_drop_combined_ = []
        self.final_features_ = []

    def _validate_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        df = df.copy()
        for col in df.columns:
            if col != self.target_col and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return df

    def _calcul_feature_importance(self, df, cache_key):
        cache_file = os.path.join(self.cache_dir, f"importance_{cache_key}.pkl")
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
            
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        spearman_corr = np.abs(X.corrwith(y, method='spearman')).fillna(0)
        rfc = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfc.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'metric1_spearman': spearman_corr,
            'metric2_mdi': rfc.feature_importances_,
            'metric3_product': spearman_corr * rfc.feature_importances_
        }).sort_values(by='metric3_product', ascending=False).reset_index(drop=True)
        
        joblib.dump(importance_df, cache_file)
        return importance_df

    def _get_cols_to_drop_intercorrelated(self, df, importance_df):
        if df.shape[1] < 2 or importance_df.empty:
            return []
        
        corr_matrix = df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        
        for col in upper.columns:
            if col in to_drop: continue
            correlated_features = upper.index[upper[col] > self.cor_val].tolist()
            if not correlated_features: continue

            all_correlated = [col] + correlated_features
            importance_subset = importance_df[importance_df['feature'].isin(all_correlated)]
            if importance_subset.empty: continue

            feature_to_keep = importance_subset.loc[importance_subset['metric3_product'].idxmax()]['feature']
            to_drop.update(f for f in all_correlated if f != feature_to_keep)
                    
        return list(to_drop)

    def _create_new_features(self, df, feature_list, epsilon=1e-6):
        new_features_df = pd.DataFrame(index=df.index)
        for feature in feature_list:
            if feature in df.columns:
                feature_values_abs = df[feature].abs()
                new_features_df[f'{feature}_pow0_5'] = np.sqrt(feature_values_abs)
                new_features_df[f'{feature}_pow2'] = df[feature] ** 2
                new_features_df[f'{feature}_log'] = np.log(feature_values_abs + epsilon)
        return new_features_df

    def fit(self, df):
        """Learns the feature engineering steps from the training data."""
        print("--- Fitting Feature Engineering Pipeline ---")
        df = self._validate_input(df)
        run_hash = f"n{self.n_select}_c{str(self.cor_val).replace('.', '')}"
        
        # 1. Base feature importance
        self.importance_df_ = self._calcul_feature_importance(df, cache_key=run_hash)
        
        # 2. Determine features to select and create
        top_m1 = self.importance_df_.nlargest(self.n_select, 'metric1_spearman')['feature']
        top_m2 = self.importance_df_.nlargest(self.n_select, 'metric2_mdi')['feature']
        self.n_select_list_ = sorted(list(set(top_m1) | set(top_m2)))
        
        top_create_m1 = self.importance_df_.nlargest(self.n_create, 'metric1_spearman')['feature']
        top_create_m2 = self.importance_df_.nlargest(self.n_create, 'metric2_mdi')['feature']
        self.n_create_list_ = sorted(list(set(top_create_m1) | set(top_create_m2)))

        # 3. Identify inter-correlated features to drop from selected list
        df_select_initial = df[self.n_select_list_]
        self.cols_to_drop_select_ = self._get_cols_to_drop_intercorrelated(df_select_initial, self.importance_df_)
        
        # 4. Create new features and identify inter-correlated ones to drop
        df_feng_initial = self._create_new_features(df, self.n_create_list_)
        if not df_feng_initial.empty:
            df_feng_with_target = df_feng_initial.join(df[[self.target_col]])
            self.feng_importance_df_ = self._calcul_feature_importance(df_feng_with_target, f"{run_hash}_feng")
            self.cols_to_drop_feng_ = self._get_cols_to_drop_intercorrelated(df_feng_initial, self.feng_importance_df_)
        
        # 5. Combine feature sets and find final correlations
        selected_feats = df_select_initial.drop(columns=self.cols_to_drop_select_, errors='ignore')
        created_feats = df_feng_initial.drop(columns=self.cols_to_drop_feng_, errors='ignore')
        df_combined_initial = pd.concat([selected_feats, created_feats], axis=1)

        if not df_combined_initial.empty:
            df_combined_with_target = df_combined_initial.join(df[[self.target_col]])
            self.combined_importance_df_ = self._calcul_feature_importance(df_combined_with_target, f"{run_hash}_combined")
            self.cols_to_drop_combined_ = self._get_cols_to_drop_intercorrelated(df_combined_initial, self.combined_importance_df_)
            
        self.final_features_ = df_combined_initial.drop(columns=self.cols_to_drop_combined_, errors='ignore').columns.tolist()
        print("--- Fitting Complete ---")
        return self

    def transform(self, df):
        """Applies the learned feature engineering steps."""
        print("--- Transforming Data ---")
        if self.final_features_ is None:
            raise RuntimeError("The pipeline has not been fitted yet. Call fit() first.")
            
        # Create all potential new features first
        df_feng_all = self._create_new_features(df, self.n_create_list_)
        
        # Combine original and new features
        df_full = pd.concat([df, df_feng_all], axis=1)
        
        # Select only the final features determined during fit
        # and ensure the target column is preserved
        missing_feats = [f for f in self.final_features_ if f not in df_full.columns]
        if missing_feats:
            raise ValueError(f"Features missing from input df: {missing_feats}")
            
        df_final = df_full[self.final_features_].copy()
        df_final[self.target_col] = df[self.target_col]
        
        print(f"Transformation complete. Final shape: {df_final.shape}")
        return df_final

# -------------------
# ML Evaluation
# -------------------
def get_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    elif scaler_name == "minmax":
        return MinMaxScaler()
    elif scaler_name == "robust":
        return RobustScaler()
    elif scaler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

def evaluate_and_log(X_train, y_train, X_test, y_test, scaler_name):
    print("\n--- Starting Model Evaluation ---")
    
    X_train_numeric = X_train.select_dtypes(include=np.number)
    X_test_numeric = X_test.select_dtypes(include=np.number)
    
    scaler = get_scaler(scaler_name)
    
    if scaler:
        print(f"Applying {scaler_name} scaler...")
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_test_scaled = scaler.transform(X_test_numeric)
    else:
        print("No scaler applied.")
        X_train_scaled = X_train_numeric.values
        X_test_scaled = X_test_numeric.values

    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
    
    mlflow.log_metric("roc_auc_test", roc_auc)
    mlflow.log_metrics({
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "final_feature_count": X_train.shape[1]
    })
    print("--- Finished Model Evaluation ---")

# -------------------
# Config and Main
# -------------------

@dataclass
class Config:
    run_name: str
    input_parquet: str
    output_dir: str
    n_select: int
    cor_val: float
    scaler: str
    cache_dir: str
    target_col: str = "TARGET"

def main(config_path: str):
    with open(config_path, 'r') as f:
        configs_data = json.load(f)
    
    base_df = None
    df_train, df_test = None, None
    current_input_path = None

    for config_dict in configs_data:
        cfg = Config(**config_dict)
        
        # Avoid reloading and splitting data if input is the same
        if current_input_path != cfg.input_parquet:
            print(f"\nLoading data from: {cfg.input_parquet}")
            base_df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
            print("Performing train-test split...")
            df_train, df_test = train_test_split(base_df, test_size=0.2, random_state=42, stratify=base_df[cfg.target_col])
            current_input_path = cfg.input_parquet

        print(f"\n=================================================")
        print(f"Starting MLflow run: {cfg.run_name}")
        print(f"=================================================")
        
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(asdict(cfg))
            
            pipeline = FeatureEngineeringPipeline(
                n_select=cfg.n_select, 
                cor_val=cfg.cor_val, 
                target_col=cfg.target_col, 
                cache_dir=cfg.cache_dir
            )
            
            # Fit on the training data ONLY
            pipeline.fit(df_train.copy())
            
            # Transform both train and test data
            train_processed = pipeline.transform(df_train.copy())
            test_processed = pipeline.transform(df_test.copy())
            
            # Save and log processed datasets
            os.makedirs(cfg.output_dir, exist_ok=True)
            train_output_path = os.path.join(cfg.output_dir, "train_processed.parquet")
            test_output_path = os.path.join(cfg.output_dir, "test_processed.parquet")
            
            train_processed.to_parquet(train_output_path, engine="pyarrow")
            test_processed.to_parquet(test_output_path, engine="pyarrow")
            mlflow.log_artifact(train_output_path, "processed_data")
            mlflow.log_artifact(test_output_path, "processed_data")
            
            # Evaluate performance on the test set
            X_train, y_train = train_processed.drop(columns=[cfg.target_col]), train_processed[cfg.target_col]
            X_test, y_test = test_processed.drop(columns=[cfg.target_col]), test_processed[cfg.target_col]
            
            evaluate_and_log(X_train, y_train, X_test, y_test, cfg.scaler)

if __name__ == "__main__":
    setup_mlflow(experiment_name="scale_and_feature_engineering", cache_dir="C:/Users/gui/Documents/credit_scoring/cache")

    parser = argparse.ArgumentParser(description="Run feature engineering pipelines from a JSON config file.")
    parser.add_argument("--config", required=True, dest="config_path", help="Path to the process_config.json file")
    
    args = parser.parse_args()
    main(args.config_path)