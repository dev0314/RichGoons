"""
add_gold_volatility.py

Apply volatility feature engineering to Gold OHLC data.

Author: RichGoons Project
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    add_xau_volatility_features
)

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(col_type):
            # add_time_features() converts "Open Time" to a real datetime64
            # column -- leave it alone, it isn't int/float and is already
            # stored compactly (8 bytes/row).
            continue

        # The original Kaggle snippet checks `col_type != object` to decide
        # whether a column is numeric. That assumption breaks on this
        # dataframe: time_features.py's "session" column comes back as
        # pandas' dedicated `string` dtype (not `object`), and
        # market_structure.py's "last_high_label"/"last_low_label" are
        # `object`. Using `is_numeric_dtype` instead catches both cases
        # correctly regardless of pandas' string-storage backend.
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "Gold 2021 - 2025"
    / "XAUUSD_5m_2021-2025.xlsx"
)

OUTPUT_PATH = "/Users/devangasaikia/Desktop/"



# =========================================================
# LOAD DATA
# =========================================================

print("Loading XAUUSD data...")

df_gold = pd.read_excel(INPUT_FILE)

print(
    f"Loaded {len(df_gold):,} rows"
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print(
    "Engineering volatility features..."
)

df_gold = add_xau_volatility_features(
    df_gold
)


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

print(
    "Reducing memory usage..."
)

df_gold = reduce_mem_usage(
    df_gold
)

# =========================================================
# SAVE
# =========================================================

print(
    "Saving feature-engineered dataset..."
)

df_gold.to_excel("/Users/devangasaikia/Desktop/XAUUSD_5m_2021-2025.xlsx", index=False)

print(
    f"Saved to:\n{OUTPUT_PATH}"
)