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

# --- Script Configuration ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Corrected data_path using a raw string for Windows compatibility
data_path = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\Enonce"
output_path = r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\df"

# --- Utility Functions ---

@contextmanager
def timer(title):
    """A timer context manager to measure the execution time of code blocks."""
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

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

# --- Data Preprocessing Functions ---

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
    df = pd.read_csv(os.path.join(path, 'application_train.csv'), nrows= num_rows)
    test_df = pd.read_csv(os.path.join(path, 'application_test.csv'), nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    
    df = pd.concat([df, test_df]).reset_index(drop=True)
    
    df = df[df['CODE_GENDER'] != 'XNA']
    
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

def bureau_and_balance(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses bureau and bureau_balance data, adding median aggregations.
    Returns the aggregated bureau dataframe.
    """
    bureau = pd.read_csv(os.path.join(path, 'bureau.csv'), nrows = num_rows)
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'median']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
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
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def previous_applications(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses previous applications data, adding median aggregations.
    Returns the aggregated previous applications dataframe.
    """
    prev = pd.read_csv(os.path.join(path, 'previous_application.csv'), nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
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
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Corrected function signature to accept 'path'
def pos_cash(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses POS_CASH_balance data, adding median aggregations.
    Returns the aggregated POS CASH balance dataframe.
    """
    pos = pd.read_csv(os.path.join(path, 'POS_CASH_balance.csv'), nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size', 'median'],
        'SK_DPD': ['max', 'mean', 'median'],
        'SK_DPD_DEF': ['max', 'mean', 'median']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
def installments_payments(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses installments_payments data, adding median aggregations.
    Returns the aggregated installments payments dataframe.
    """
    ins = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
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
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

def credit_card_balance(path, num_rows = None, nan_as_category = True):
    """
    Loads and preprocesses credit_card_balance data, adding median aggregations.
    Returns the aggregated credit card balance dataframe.
    """
    cc = pd.read_csv(os.path.join(path, 'credit_card_balance.csv'), nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var', 'median'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# --- Main Execution Function ---

def main(data_path, output_path, debug=False):
    """
    Main function to run the data ingestion and preprocessing pipeline.
    
    Parameters:
    - data_path (str): The path to the directory containing the raw data CSV files.
    - output_path (str): The path to the directory where the output parquet file will be stored.
    - debug (bool): If True, runs in debug mode with a small number of rows.
    """
    num_rows = 10000 if debug else None
    
    with timer("Process application train and test"):
        df = application_train_test(data_path, num_rows)
        print("Application df shape:", df.shape)
    
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

    # --- ADDED: Save the final DataFrame ---
    with timer("Save final dataframe to parquet file"):
        # Define the full output file path
        output_file = os.path.join(output_path, "df_global.parquet")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save the dataframe
        df.to_parquet(output_file)
        
        print(f"Final DataFrame shape: {df.shape}")
        print(f"Successfully saved df_global.parquet to {output_file}")


# --- ADDED: Script entry point ---
if __name__ == "__main__":
    # Set debug=True for a quick run with a subset of data
    main(data_path=data_path, output_path=output_path, debug=False)