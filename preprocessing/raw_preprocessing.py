import os
import gc
import time
from contextlib import contextmanager
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(path, num_rows=None, nan_as_category=False):
    df = pd.read_csv(os.path.join(path, 'application_train.csv'), nrows=num_rows)
    test_df = pd.read_csv(os.path.join(path, 'application_test.csv'), nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = pd.concat([df, test_df]).reset_index(drop=True)
    df = df[df['CODE_GENDER'] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        if bin_feature in df.columns:
            df[bin_feature], _ = pd.factorize(df[bin_feature])
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # safe feature creation (guard missing cols)
    def safe_div(a, b):
        return a / b if a is not None and b is not None else np.nan
    if {'DAYS_EMPLOYED', 'DAYS_BIRTH'}.issubset(df.columns):
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    if {'AMT_INCOME_TOTAL', 'AMT_CREDIT'}.issubset(df.columns):
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    if {'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'}.issubset(df.columns):
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(df.columns):
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    if {'AMT_ANNUITY', 'AMT_CREDIT'}.issubset(df.columns):
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

def bureau_and_balance(path, num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(os.path.join(path, 'bureau.csv'), nrows=num_rows)
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'median']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True, errors='ignore')
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
        'CNT_CREDIT_PROLONG': ['sum']
    }
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    active = bureau[bureau.get('CREDIT_ACTIVE_Active', 0) == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations) if not active.empty else pd.DataFrame()
    if not active_agg.empty:
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    closed = bureau[bureau.get('CREDIT_ACTIVE_Closed', 0) == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations) if not closed.empty else pd.DataFrame()
    if not closed_agg.empty:
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def previous_applications(path, num_rows=None, nan_as_category=True):
    prev = pd.read_csv(os.path.join(path, 'previous_application.csv'), nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=nan_as_category)
    for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        if c in prev.columns:
            prev[c].replace(365243, np.nan, inplace=True)
    if {'AMT_APPLICATION', 'AMT_CREDIT'}.issubset(prev.columns):
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
    cat_aggregations = {cat: ['mean'] for cat in cat_cols}
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations}) if not prev.empty else pd.DataFrame()
    if not prev_agg.empty:
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        approved = prev[prev.get('NAME_CONTRACT_STATUS_Approved', 0) == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations) if not approved.empty else pd.DataFrame()
        if not approved_agg.empty:
            approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
            prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        refused = prev[prev.get('NAME_CONTRACT_STATUS_Refused', 0) == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations) if not refused.empty else pd.DataFrame()
        if not refused_agg.empty:
            refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
            prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
    return prev_agg

def pos_cash(path, num_rows=None, nan_as_category=True):
    pos = pd.read_csv(os.path.join(path, 'POS_CASH_balance.csv'), nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category)
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

def installments_payments(path, num_rows=None, nan_as_category=True):
    ins = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=nan_as_category)
    if {'AMT_PAYMENT', 'AMT_INSTALMENT'}.issubset(ins.columns):
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins.get('DAYS_ENTRY_PAYMENT', 0) - ins.get('DAYS_INSTALMENT', 0)
    ins['DBD'] = ins.get('DAYS_INSTALMENT', 0) - ins.get('DAYS_ENTRY_PAYMENT', 0)
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

def credit_card_balance(path, num_rows=None, nan_as_category=True):
    cc = pd.read_csv(os.path.join(path, 'credit_card_balance.csv'), nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True, errors='ignore')
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var', 'median'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def build_df_global(raw_path, output_path, num_rows=None):
    print("Building df_global from raw CSVs in:", raw_path)
    df = application_train_test(raw_path, num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(raw_path, num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau; gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(raw_path, num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev; gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(raw_path, num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos; gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(raw_path, num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins; gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(raw_path, num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc; gc.collect()
    # Save df_global
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow')
    print(f"Saved df_global to {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-path", default=r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\Enonce", help="Folder with raw CSV files")
    p.add_argument("--output", default=r"C:\Users\gui\Documents\OpenClassrooms\Projet 7\df\df_global.parquet", help="Output parquet path")
    p.add_argument("--nrows", type=int, default=None)
    args = p.parse_args()
    build_df_global(args.raw_path, args.output, num_rows=args.nrows)