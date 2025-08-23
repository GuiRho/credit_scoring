import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow

# -------------------
# Feature Engineering Pipeline
# -------------------

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
            
        spearman_corr = np.abs(X.corrwith(y, method='spearman')).fillna(0)
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

# -------------------
# Config and Main
# -------------------

@dataclass
class Config:
    input_parquet: str
    output_parquet: str
    n_select: int = 50
    cor_val: float = 0.7
    cache_dir: Optional[str] = "cache"
    target_col: str = "TARGET"

def main(cfg: Config):
    print("Loading data from:", cfg.input_parquet)
    df = pd.read_parquet(cfg.input_parquet, engine='pyarrow')
    
    pipeline = FeatureEngineeringPipeline(
        n_select=cfg.n_select, 
        cor_val=cfg.cor_val, 
        target_col=cfg.target_col, 
        cache_dir=cfg.cache_dir
    )
    
    _, _, df_final = pipeline.run(df.copy())
    
    print("Saving final data to:", cfg.output_parquet)
    df_final.to_parquet(cfg.output_parquet, engine="pyarrow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and run feature engineering.")
    parser.add_argument("--input", required=True, dest="input_parquet", help="Input parquet path")
    parser.add_argument("--output", required=True, dest="output_parquet", help="Output processed parquet path")
    parser.add_argument("--n_select", type=int, default=50)
    parser.add_argument("--cor_val", type=float, default=0.7)
    parser.add_argument("--cache-dir", dest="cache_dir", default="cache")
    parser.add_argument("--target", dest="target_col", default="TARGET")
    
    args = parser.parse_args()
    
    config = Config(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        n_select=args.n_select,
        cor_val=args.cor_val,
        cache_dir=args.cache_dir,
        target_col=args.target_col
    )
    
    main(config)
