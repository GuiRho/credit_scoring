import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import warnings
from itertools import combinations
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

@contextmanager
def timer(title):
    """A timer context manager to measure the execution time of code blocks."""
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    """
    Performs one-hot encoding on categorical columns of a dataframe.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - nan_as_category (bool): Whether to create a separate column for NaN values.
    
    Returns:
    - pd.DataFrame: The dataframe with one-hot encoded features.
    - list: A list of the new column names.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(path, num_rows = None, nan_as_category = False):
    """
    Loads and preprocesses the main application data.
    
    Parameters:
    - path (str): The path to the directory containing the CSV files.
    - num_rows (int, optional): The number of rows to load from the CSV files.
    - nan_as_category (bool): Whether to treat NaNs as a category during one-hot encoding.

    Returns:
    - pd.DataFrame: The preprocessed application dataframe.
    """
    # Read data and merge
    df = pd.read_csv(f'{path}application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(f'{path}application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    
    # Use pd.concat instead of the deprecated .append()
    df = pd.concat([df, test_df]).reset_index(drop=True) # drop=True to avoid creating a new 'index' column
    
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses bureau and bureau_balance data, adding median aggregations.
    
    Parameters are similar to the function above.
    Returns the aggregated bureau dataframe.
    """
    bureau = pd.read_csv(f'{path}bureau.csv', nrows = num_rows)
    bb = pd.read_csv(f'{path}bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'median']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var', 'median'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean', 'median'],
        'DAYS_CREDIT_UPDATE': ['mean', 'median'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'median'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'median'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', 'median'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum', 'median'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'median'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum', 'median'],
        'AMT_ANNUITY': ['max', 'mean', 'median'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum', 'median']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses previous applications data, adding median aggregations.
    Parameters are similar to the function above.
    Returns the aggregated previous applications dataframe.
    """
    prev = pd.read_csv(f'{path}previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean', 'median'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'median'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'median'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var', 'median'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'median'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean', 'median'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean', 'median'],
        'DAYS_DECISION': ['min', 'max', 'mean', 'median'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses POS_CASH_balance data, adding median aggregations.
    Parameters are similar to the function above.
    Returns the aggregated POS CASH balance dataframe.
    """
    pos = pd.read_csv(f'{path}POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size', 'median'],
        'SK_DPD': ['max', 'mean', 'median'],
        'SK_DPD_DEF': ['max', 'mean', 'median']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses installments_payments data, adding median aggregations.
    Parameters are similar to the function above.
    Returns the aggregated installments payments dataframe.
    """
    ins = pd.read_csv(f'{path}installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'median'],
        'DBD': ['max', 'mean', 'sum', 'median'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var', 'median'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var', 'median'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'median'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'median'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'median']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses credit_card_balance data, adding median aggregations.
    Parameters are similar to the function above.
    Returns the aggregated credit card balance dataframe.
    """
    cc = pd.read_csv(f'{path}credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var', 'median'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def clean_and_impute_data(df, target_col, completeness=85, impute='median', verbose=True):
    df_processed = df.copy()
    initial_cols = df_processed.shape[1]
    initial_rows = df_processed.shape[0]
    
    # Drop columns with completeness below threshold
    col_completeness = (1 - df_processed.isnull().sum() / len(df_processed)) * 100
    cols_to_drop_completeness = col_completeness[col_completeness < completeness].index.tolist()
    if cols_to_drop_completeness:
        df_processed.drop(columns=cols_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(cols_to_drop_completeness)} columns due to completeness < {completeness}%")

    # Drop rows with completeness below threshold (using 50% of column completeness threshold)
    row_completeness = (1 - df_processed.isnull().sum(axis=1) / df_processed.shape[1]) * 100
    rows_to_drop_completeness = df_processed[row_completeness < completeness*0.5].index.tolist()
    if rows_to_drop_completeness:
        df_processed.drop(index=rows_to_drop_completeness, inplace=True)
        if verbose:
            print(f"Dropped {len(rows_to_drop_completeness)} rows due to completeness < {completeness*0.5}%")

    # Impute numerical columns
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if impute == 'median':
        impute_value = df_processed[numerical_cols].median()
    elif impute == 'mean':
        impute_value = df_processed[numerical_cols].mean()
    elif impute == 'zero':
        impute_value = 0
    else:
        raise ValueError(f"Méthode d'imputation non reconnue : {impute}")
    df_processed[numerical_cols] = df_processed[numerical_cols].fillna(impute_value)
    
    # Drop rows where target is NaN (if target_col exists and has NaNs)
    if target_col in df_processed.columns:
        df_processed.dropna(subset=[target_col], inplace=True)
        df_processed[target_col] = df_processed[target_col].astype(int)

    if verbose:
        print(f"Original shape: ({initial_rows}, {initial_cols})")
        print(f"Processed shape: {df_processed.shape}")
    return df_processed

def remove_percent_outliers_2sides(df, percent):
    # Create a copy of the original DataFrame
    df_cleaned = df.copy()

    # Convert each column to numeric, forcing errors to NaN
    for feat in df_cleaned.columns:
        df_cleaned[feat] = pd.to_numeric(df_cleaned[feat], errors='coerce')

    # Drop rows with NaN values that resulted from coercion
    df_cleaned = df_cleaned.dropna()

    # Ensure that the DataFrame is not empty after dropping NaN values
    if df_cleaned.empty:
        print("DataFrame is empty after dropping NaN values.")
        return df_cleaned

    # Collect outlier indices
    all_outlier_indices = []
    lower_quantile = percent / 100
    upper_quantile = (100 - percent) / 100

    for feat in df_cleaned.select_dtypes(include=[np.number]).columns:
        # Calculate the thresholds for outlier detection
        lower_val = df_cleaned[feat].quantile(lower_quantile)
        upper_val = df_cleaned[feat].quantile(upper_quantile)

        # Identify outliers on both sides
        outlier_rows = df_cleaned[(df_cleaned[feat] < lower_val) | (df_cleaned[feat] > upper_val)]

        # Collect outlier indices
        outlier_indices = outlier_rows.index.tolist()
        all_outlier_indices.extend(outlier_indices)

    # Remove duplicate indices
    all_outlier_indices = list(set(all_outlier_indices))
    print(f"Number of total outliers = {len(all_outlier_indices)}")

    # Remove outliers from the DataFrame
    df_cleaned = df_cleaned.drop(index=all_outlier_indices)

    return df_cleaned

class FeatureEngineeringPipeline:
    """A simplified class for feature selection and engineering on a single DataFrame."""
    def __init__(self, n_select: int = 50, cor_val: float = 0.7, target_col: str = 'TARGET'):
        """
        Initialize the pipeline with fixed parameters.
        
        Args:
            n_select (int): Number of top features to select (default: 50).
            cor_val (float): Correlation threshold for dropping features (default: 0.7).
            target_col (str): Name of the target column (default: 'TARGET').
        """
        self.n_select = n_select
        self.n_create = max(2, int(np.sqrt(n_select)))
        self.cor_val = cor_val
        self.target_col = target_col
        self.importance_df = None
        self.feng_importance_df = None
        self.combined_importance_df = None

    def _validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame and convert boolean columns to numeric."""
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

    def _calcul_feature_importance(self, df: pd.DataFrame, cache_key: str):
        """Calculate feature importance with caching."""
        if df.empty or df.shape[1] <= 1:
            return pd.DataFrame(columns=['feature', 'metric1_spearman', 'metric2_mdi', 'metric3_product'])

        cache_dir = "c:\\Users\\gui\\Documents\\credit_scoring\\cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"importance_{cache_key}.pkl")

        if os.path.exists(cache_file):
            cached_df = joblib.load(cache_file)
            # Check if the cached features match the current dataframe's features
            # Exclude target column from comparison
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
        """Select top features based on importance metrics."""
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

    def _create_new_features(self, df: pd.DataFrame, feature_list: list, epsilon: float = 1e-6):
        """Create new features from a subset of features."""
        if not feature_list:
            return pd.DataFrame(index=df.index)
        new_features_list = []
        for feature in feature_list:
            if feature in df.columns:
                feature_values_abs = df[feature].abs()
                new_features_list.append(pd.Series(np.sqrt(feature_values_abs), name=f'{feature}_pow0_5', index=df.index))
                new_features_list.append(pd.Series(df[feature] ** 2, name=f'{feature}_pow2', index=df.index))
                new_features_list.append(pd.Series(np.log(feature_values_abs + epsilon), name=f'{feature}_log', index=df.index))
        for f1, f2 in combinations(feature_list, 2):
            if f1 in df.columns and f2 in df.columns:
                new_features_list.append(pd.Series(df[f1] + df[f2], name=f'{f1}_plus_{f2}', index=df.index))
                new_features_list.append(pd.Series(df[f1] * df[f2], name=f'{f1}_times_{f2}', index=df.index))
        if not new_features_list:
            return pd.DataFrame(index=df.index) # Return empty if no features were created
        return pd.concat(new_features_list, axis=1)

    def _drop_intercorrelated(self, df: pd.DataFrame, importance_df: pd.DataFrame):
        """Drop inter-correlated features based on importance."""
        if df.empty or importance_df.empty or len(df.columns) < 2:
            return df
        
        # Ensure all columns are numeric before calculating correlation
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return df # No numeric columns to correlate or only one

        corr_matrix = numeric_df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            correlated_features = upper.index[upper[col] > self.cor_val].tolist()
            if correlated_features:
                all_correlated = [col] + correlated_features
                # Filter importance_df to only include features present in all_correlated and df.columns
                importance_subset = importance_df[importance_df['feature'].isin(all_correlated)]
                importance_subset = importance_subset[importance_subset['feature'].isin(df.columns)]

                if not importance_subset.empty:
                    # Ensure 'metric3_product' is numeric and handle potential NaNs
                    importance_subset = importance_subset.copy()
                    importance_subset['metric3_product'] = pd.to_numeric(importance_subset['metric3_product'], errors='coerce').fillna(0)
                    
                    feature_to_keep = importance_subset.loc[importance_subset['metric3_product'].idxmax()]['feature']
                    to_drop.update(f for f in all_correlated if f != feature_to_keep and f in df.columns)
        return df.drop(columns=list(to_drop), errors='ignore')

    def run(self, df: pd.DataFrame):
        """
        Run the feature engineering pipeline and return three DataFrames with TARGET column.
        
        Args:
            df (pd.DataFrame): Input DataFrame with numeric/boolean features and binary TARGET column.
        
        Returns:
            tuple: (df_select, df_feng, df_combined) DataFrames with TARGET column.
        """
        print("Step 1: Validating input DataFrame...")
        df = self._validate_input(df)

        print("Step 2: Calculating feature importance...")
        run_hash = f"n{self.n_select}_c{str(self.cor_val).replace('.', '')}"
        self.importance_df = self._calcul_feature_importance(df, cache_key=run_hash)

        print("Step 3: Selecting feature lists...")
        n_select_list, n_create_list = self._top_feature_selection()

        print("Step 4: Processing 'df_select'...")
        # Ensure df[n_select_list] only selects columns that actually exist in df
        existing_n_select_list = [col for col in n_select_list if col in df.columns]
        df_select = self._drop_intercorrelated(df[existing_n_select_list], self.importance_df)
        df_select = df_select.join(df[[self.target_col]])  # Add TARGET column

        print("Step 5: Processing 'df_feng'...")
        existing_n_create_list = [col for col in n_create_list if col in df.columns]
        df_feng_initial = self._create_new_features(df, existing_n_create_list)
        
        # Ensure df_feng_initial is not empty before calculating importance
        if not df_feng_initial.empty:
            self.feng_importance_df = self._calcul_feature_importance(
                df_feng_initial.join(df[[self.target_col]]), cache_key=f"{run_hash}_feng"
            )
            df_feng = self._drop_intercorrelated(df_feng_initial, self.feng_importance_df)
        else:
            df_feng = pd.DataFrame(index=df.index) # Return empty if no features were created
        df_feng = df_feng.join(df[[self.target_col]])  # Add TARGET column

        print("Step 6: Processing 'df_combined'...")
        # Ensure columns exist before dropping
        cols_to_drop_select = [self.target_col] if self.target_col in df_select.columns else []
        cols_to_drop_feng = [self.target_col] if self.target_col in df_feng.columns else []

        df_combined_initial = pd.concat([
            df_select.drop(columns=cols_to_drop_select, errors='ignore'), 
            df_feng.drop(columns=cols_to_drop_feng, errors='ignore')
        ], axis=1)

        if not df_combined_initial.empty:
            self.combined_importance_df = self._calcul_feature_importance(
                df_combined_initial.join(df[[self.target_col]]), cache_key=f"{run_hash}_combined"
            )
            df_combined = self._drop_intercorrelated(df_combined_initial, self.combined_importance_df)
        else:
            df_combined = pd.DataFrame(index=df.index)
        df_combined = df_combined.join(df[[self.target_col]])  # Add TARGET column

        print("Feature engineering complete.")
        return df_select, df_feng, df_combined

def main(data_path, debug=False):
    """
    Main function to run the data ingestion and preprocessing pipeline.
    
    Parameters:
    - data_path (str): The path to the directory containing the raw data CSV files.
    - debug (bool): If True, runs in debug mode with a small number of rows.
    
    Returns:
    - pd.DataFrame: The final, merged and cleaned dataframe.
    """
    num_rows = 10000 if debug else None
    df = application_train_test(data_path, num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(data_path, num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(data_path, num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(data_path, num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(data_path, num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(data_path, num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    
    # Apply cleaning and imputation after initial feature engineering
    df_cleaned = clean_and_impute_data(df, target_col='TARGET', completeness=85, impute='median')
    
    # Apply outlier removal
    df_cleaned = remove_percent_outliers_2sides(df_cleaned, percent=1)

    # Apply feature engineering pipeline
    print("\n--- Running Feature Engineering Pipeline ---")
    pipeline = FeatureEngineeringPipeline(n_select=50, cor_val=0.7)
    _, _, df_final = pipeline.run(df_cleaned.copy())

    return df_final

# This file has been replaced by a thin wrapper. Heavy raw CSV preprocessing is now in:
#   preprocessing/raw_preprocessing.py
# Processing of the immutable df_global.parquet is in:
#   processing/process_df_global.py
# Run either script directly:
#   python -m preprocessing.raw_preprocessing --raw-path "<raw csv folder>" --output "<df_global.parquet>"
#   python -m processing.process_df_global --input "<df_global.parquet>" --output "<processed.parquet>"
if __name__ == "__main__":
    import sys
    print("This repository now separates raw preprocessing and df_global processing.")
    print("See preprocessing/raw_preprocessing.py and processing/process_df_global.py")
    print("Examples:")
    print("  python -m preprocessing.raw_preprocessing --raw-path \"C:/.../Enonce\" --output \"C:/.../df/df_global.parquet\"")
    print("  python -m processing.process_df_global --input \"C:/.../df/df_global.parquet\" --output \"./df_final.parquet\"")
    sys.exit(0)