import os, json, argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib, mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
        self.importance_df_: Optional[pd.DataFrame] = None
        self.feng_importance_df_: Optional[pd.DataFrame] = None
        self.combined_importance_df_: Optional[pd.DataFrame] = None
        self.n_select_list_: List[str] = []
        self.n_create_list_: List[str] = []
        self.cols_to_drop_select_: List[str] = []
        self.cols_to_drop_feng_: List[str] = []
        self.cols_to_drop_combined_: List[str] = []
        self.final_features_: List[str] = []

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty DataFrame")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        df = df.copy()
        for col in df.columns:
            if col != self.target_col and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return df

    def _calculate_feature_importance(self, df: pd.DataFrame, cache_key: str) -> pd.DataFrame:
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

    def _get_cols_to_drop_intercorrelated(self, df: pd.DataFrame, importance_df: pd.DataFrame) -> List[str]:
        if df.shape[1] < 2 or importance_df.empty:
            return []
        corr_matrix = df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop: set = set()
        for col in upper.columns:
            correlated_features = upper.index[upper[col] > self.cor_val].tolist()
            if not correlated_features:
                continue
            all_correlated = [col] + correlated_features
            importance_subset = importance_df[importance_df['feature'].isin(all_correlated)]
            if not importance_subset.empty:
                feature_to_keep = importance_subset.loc[importance_subset['metric3_product'].idxmax()]['feature']
                to_drop.update(f for f in all_correlated if f != feature_to_keep)
        return list(to_drop)

    def _create_new_features(self, df: pd.DataFrame, feature_list: List[str], epsilon: float = 1e-6) -> pd.DataFrame:
        new_features_df = pd.DataFrame(index=df.index)
        for feature in feature_list:
            if feature in df.columns:
                f_abs = df[feature].abs()
                new_features_df[f'{feature}_pow0_5'] = np.sqrt(f_abs)
                new_features_df[f'{feature}_pow2'] = df[feature] ** 2
                new_features_df[f'{feature}_log'] = np.log(f_abs + epsilon)
        return new_features_df

    def fit(self, df: pd.DataFrame):
        df = self._validate_input(df)
        run_hash = f"n{self.n_select}_c{str(self.cor_val).replace('.', '')}"
        self.importance_df_ = self._calculate_feature_importance(df, cache_key=run_hash)
        top_m1 = self.importance_df_.nlargest(self.n_select, 'metric1_spearman')['feature']
        top_m2 = self.importance_df_.nlargest(self.n_select, 'metric2_mdi')['feature']
        self.n_select_list_ = sorted(list(set(top_m1) | set(top_m2)))
        top_c1 = self.importance_df_.nlargest(self.n_create, 'metric1_spearman')['feature']
        top_c2 = self.importance_df_.nlargest(self.n_create, 'metric2_mdi')['feature']
        self.n_create_list_ = sorted(list(set(top_c1) | set(top_c2)))
        self.cols_to_drop_select_ = self._get_cols_to_drop_intercorrelated(df[self.n_select_list_], self.importance_df_)
        df_feng_initial = self._create_new_features(df, self.n_create_list_)
        if not df_feng_initial.empty:
            df_feng_with_target = df_feng_initial.join(df[[self.target_col]])
            self.feng_importance_df_ = self._calculate_feature_importance(df_feng_with_target, f"{run_hash}_feng")
            self.cols_to_drop_feng_ = self._get_cols_to_drop_intercorrelated(df_feng_initial, self.feng_importance_df_)
        selected = df[self.n_select_list_].drop(columns=self.cols_to_drop_select_, errors='ignore')
        created = df_feng_initial.drop(columns=self.cols_to_drop_feng_, errors='ignore')
        df_combined = pd.concat([selected, created], axis=1)
        if not df_combined.empty:
            self.combined_importance_df_ = self._calculate_feature_importance(
                df_combined.join(df[[self.target_col]]), f"{run_hash}_combined")
            self.cols_to_drop_combined_ = self._get_cols_to_drop_intercorrelated(df_combined, self.combined_importance_df_)
        self.final_features_ = df_combined.drop(columns=self.cols_to_drop_combined_, errors='ignore').columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.final_features_:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        df_full = pd.concat([df, self._create_new_features(df, self.n_create_list_)], axis=1)
        missing = [f for f in self.final_features_ if f not in df_full.columns]
        if missing:
            raise ValueError(f"Features missing from input: {missing}")
        df_final = df_full[self.final_features_].copy()
        df_final[self.target_col] = df[self.target_col]
        print(f"Transformation complete. Final shape: {df_final.shape}")
        return df_final

def get_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    elif scaler_name == "minmax":
        return MinMaxScaler()
    elif scaler_name == "robust":
        return RobustScaler()
    elif scaler_name == "none":
        return None
    raise ValueError(f"Unknown scaler: {scaler_name}")

def evaluate_and_log(X_train, y_train, X_test, y_test, scaler_name):
    print("\n--- Starting Model Evaluation ---")
    scaler = get_scaler(scaler_name)
    X_tr = X_train.select_dtypes(include=np.number)
    X_te = X_test.select_dtypes(include=np.number)
    if scaler:
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr.values, X_te.values
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    model.fit(X_tr_s, y_train)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_te_s)[:, 1])
    print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
    mlflow.log_metric("roc_auc_test", roc_auc)
    mlflow.log_metrics({"train_rows": len(X_train), "test_rows": len(X_test), "final_feature_count": X_train.shape[1]})
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
    base_df = df_train = df_test = current_input_path = None
    for config_dict in configs_data:
        cfg = Config(**config_dict)
        if current_input_path != cfg.input_parquet:
            print(f"\nLoading data from: {cfg.input_parquet}")
            base_df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
            df_train, df_test = train_test_split(base_df, test_size=0.2, random_state=42, stratify=base_df[cfg.target_col])
            current_input_path = cfg.input_parquet
        print(f"\n--- MLflow run: {cfg.run_name} ---")
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(asdict(cfg))
            pipeline = FeatureEngineeringPipeline(n_select=cfg.n_select, cor_val=cfg.cor_val, target_col=cfg.target_col, cache_dir=cfg.cache_dir)
            pipeline.fit(df_train.copy())
            train_p = pipeline.transform(df_train.copy())
            test_p = pipeline.transform(df_test.copy())
            os.makedirs(cfg.output_dir, exist_ok=True)
            for tag, df_out in [("train", train_p), ("test", test_p)]:
                path = os.path.join(cfg.output_dir, f"{tag}_processed.parquet")
                df_out.to_parquet(path, engine="pyarrow"); mlflow.log_artifact(path, "processed_data")
            X_train, y_train = train_p.drop(columns=[cfg.target_col]), train_p[cfg.target_col]
            X_test, y_test = test_p.drop(columns=[cfg.target_col]), test_p[cfg.target_col]
            evaluate_and_log(X_train, y_train, X_test, y_test, cfg.scaler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering pipelines from a JSON config file.")
    parser.add_argument("--config", required=True, dest="config_path")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment", default="process", help="MLflow experiment name")
    args = parser.parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    main(args.config_path)
