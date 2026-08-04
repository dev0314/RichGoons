"""
price_features.py

Feature engineering for price-based features.

Author: RichGoons Project

Required Columns
----------------
Open
High
Low
Close
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Validate required columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate price-derived features.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    _validate_columns(df)

    df = df.copy()

    # =====================================================
    # PRICE LEVELS
    # =====================================================

    df["hl2"] = (
        df["High"] + df["Low"]
    ) / 2

    df["hlc3"] = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    df["ohlc4"] = (
        df["Open"] +
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 4

    df["typical_price"] = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    df["median_price"] = (
        df["High"] +
        df["Low"]
    ) / 2

    # =====================================================
    # ABSOLUTE PRICE CHANGES
    # =====================================================

    df["price_change"] = (
        df["Close"] - df["Open"]
    )

    df["high_low_diff"] = (
        df["High"] - df["Low"]
    )

    df["close_open_diff"] = (
        df["Close"] - df["Open"]
    )

    df["high_close_diff"] = (
        df["High"] - df["Close"]
    )

    df["close_low_diff"] = (
        df["Close"] - df["Low"]
    )

    # =====================================================
    # PERCENT RETURNS
    # =====================================================

    df["return_1"] = (
        df["Close"]
        .pct_change(1)
    )

    df["return_3"] = (
        df["Close"]
        .pct_change(3)
    )

    df["return_6"] = (
        df["Close"]
        .pct_change(6)
    )

    df["return_12"] = (
        df["Close"]
        .pct_change(12)
    )

    df["return_24"] = (
        df["Close"]
        .pct_change(24)
    )

    # =====================================================
    # LOG RETURNS
    # =====================================================

    df["log_return_1"] = np.log(
        df["Close"] /
        df["Close"].shift(1)
    )

    df["log_return_3"] = np.log(
        df["Close"] /
        df["Close"].shift(3)
    )

    # =====================================================
    # PRICE RATIOS
    # =====================================================

    df["close_open_ratio"] = (
        df["Close"] /
        df["Open"]
    )

    df["high_low_ratio"] = (
        df["High"] /
        df["Low"]
    )

    df["close_high_ratio"] = (
        df["Close"] /
        df["High"]
    )

    df["close_low_ratio"] = (
        df["Close"] /
        df["Low"]
    )

    # =====================================================
    # NORMALIZED PRICE LOCATION
    # =====================================================

    candle_range = (
        df["High"] - df["Low"]
    ).replace(0, np.nan)

    df["price_position"] = (
        (df["Close"] - df["Low"]) /
        candle_range
    )

    # =====================================================
    # GAP FEATURES
    # =====================================================

    df["gap"] = (
        df["Open"] -
        df["Close"].shift(1)
    )

    df["gap_pct"] = (
        df["gap"] /
        df["Close"].shift(1)
    )

    # =====================================================
    # DISTANCE FEATURES
    # =====================================================

    df["dist_high"] = (
        df["High"] -
        df["Close"]
    )

    df["dist_low"] = (
        df["Close"] -
        df["Low"]
    )

    df["dist_open"] = (
        df["Close"] -
        df["Open"]
    )

    return df