"""
candle_features.py

Feature engineering for candlestick-based features.

Author: RichGoons Project

Required Columns:
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
    Ensure required columns exist.
    """

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds candlestick-derived features.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with new candle features.
    """

    _validate_columns(df)

    df = df.copy()

    # ----------------------------------------------------
    # Candle Body
    # ----------------------------------------------------

    df["body"] = (df["Close"] - df["Open"]).abs()

    df["body_signed"] = (
        df["Close"] - df["Open"]
    )

    # ----------------------------------------------------
    # Candle Direction
    # ----------------------------------------------------

    df["bullish"] = (
        df["Close"] > df["Open"]
    ).astype(int)

    df["bearish"] = (
        df["Close"] < df["Open"]
    ).astype(int)

    df["doji"] = (
        df["Close"] == df["Open"]
    ).astype(int)

    # ----------------------------------------------------
    # Candle Range
    # ----------------------------------------------------

    df["range"] = (
        df["High"] - df["Low"]
    )

    # Avoid divide-by-zero
    safe_range = df["range"].replace(0, np.nan)

    # ----------------------------------------------------
    # Upper Wick
    # ----------------------------------------------------

    df["upper_wick"] = (
        df["High"]
        - df[["Open", "Close"]].max(axis=1)
    )

    # ----------------------------------------------------
    # Lower Wick
    # ----------------------------------------------------

    df["lower_wick"] = (
        df[["Open", "Close"]].min(axis=1)
        - df["Low"]
    )

    # ----------------------------------------------------
    # Ratios
    # ----------------------------------------------------

    df["body_ratio"] = (
        df["body"] / safe_range
    )

    df["upper_wick_ratio"] = (
        df["upper_wick"] / safe_range
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] / safe_range
    )

    # ----------------------------------------------------
    # Body Position
    # ----------------------------------------------------

    df["close_position"] = (
        (df["Close"] - df["Low"]) / safe_range
    )

    df["open_position"] = (
        (df["Open"] - df["Low"]) / safe_range
    )

    # ----------------------------------------------------
    # Wick Dominance
    # ----------------------------------------------------

    df["upper_wick_dominant"] = (
        df["upper_wick"] > df["body"]
    ).astype(int)

    df["lower_wick_dominant"] = (
        df["lower_wick"] > df["body"]
    ).astype(int)

    # ----------------------------------------------------
    # Marubozu
    # Very small wicks
    # ----------------------------------------------------

    df["marubozu"] = (
        (df["upper_wick_ratio"] < 0.05)
        &
        (df["lower_wick_ratio"] < 0.05)
    ).astype(int)

    # ----------------------------------------------------
    # Hammer
    # ----------------------------------------------------

    df["hammer"] = (
        (df["lower_wick"] >= 2 * df["body"])
        &
        (df["upper_wick"] <= df["body"])
    ).astype(int)

    # ----------------------------------------------------
    # Shooting Star
    # ----------------------------------------------------

    df["shooting_star"] = (
        (df["upper_wick"] >= 2 * df["body"])
        &
        (df["lower_wick"] <= df["body"])
    ).astype(int)

    # ----------------------------------------------------
    # Spinning Top
    # ----------------------------------------------------

    df["spinning_top"] = (
        (df["body_ratio"] < 0.30)
        &
        (df["upper_wick_ratio"] > 0.25)
        &
        (df["lower_wick_ratio"] > 0.25)
    ).astype(int)

    return df