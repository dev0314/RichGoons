import numpy as np
import pandas as pd
import os
from features import FeaturePipeline


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


df_23 = pd.read_csv("BTCUSDT_5m_2023.csv")
df_24 = pd.read_csv("BTCUSDT_5m_2024.csv")
df_25 = pd.read_csv("BTCUSDT_5m_2025.csv")

df = pd.concat([df_23, df_24, df_25], ignore_index=True)
df = df.drop_duplicates()

pipeline = FeaturePipeline()
enriched_df = pipeline.fit_transform(df)

enriched_df = reduce_mem_usage(enriched_df)

desktop_path = os.path.expanduser("~/Desktop/BTCUSDT_5m_2023-2025_features_compressed.parquet")
enriched_df.to_parquet(desktop_path, index=False)
#enriched_df.to_parquet("BTCUSDT_5m_2023-2025_features_compressed.parquet", index=False, compression="zstd")