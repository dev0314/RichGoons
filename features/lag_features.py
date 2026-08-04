"""
lag_features.py

Creates lag-based features for time series datasets.

Author: RichGoons Project
"""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] | range = range(1, 6),
    create_difference: bool = True,
    create_pct_change: bool = True,
    create_ratio: bool = False,
) -> pd.DataFrame:
    """
    Generate lag-based features.

    Parameters
    ----------
    df : pd.DataFrame

    columns : list[str]
        Columns to generate lag features for.

    lags : list[int]
        Lag periods.

    create_difference : bool
        Create difference features.

    create_pct_change : bool
        Create percentage change features.

    create_ratio : bool
        Create ratio features.

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    for column in columns:

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        for lag in lags:

            lag_col = f"{column.lower()}_lag_{lag}"

            df[lag_col] = df[column].shift(lag)

            # ------------------------------------------
            # Difference
            # ------------------------------------------

            if create_difference:

                df[f"{column.lower()}_diff_{lag}"] = (
                    df[column] -
                    df[lag_col]
                )

            # ------------------------------------------
            # Percentage Change
            # ------------------------------------------

            if create_pct_change:

                df[f"{column.lower()}_pct_change_{lag}"] = (
                    (
                        df[column] -
                        df[lag_col]
                    )
                    /
                    df[lag_col]
                )

            # ------------------------------------------
            # Ratio
            # ------------------------------------------

            if create_ratio:

                df[f"{column.lower()}_ratio_{lag}"] = (
                    df[column] /
                    df[lag_col]
                )

    return df