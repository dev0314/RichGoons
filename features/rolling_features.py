"""
rolling_features.py

Rolling window feature engineering.

Produces three families of features per configured column:

1. Statistical Features   -> mean, std, min, max, median, skew, kurtosis, sum
2. Relative Features       -> distance from mean, ratio to mean, z-score, percentile rank
3. Trend Features          -> mean slope, mean crossover, std ratio (vol expansion),
                              rolling range, rolling momentum

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers (cached rolling stats so relative/trend features don't
# recompute the same rolling window twice)
# ---------------------------------------------------------------------------

def _roll(df: pd.DataFrame, column: str, window: int, stat: str, cache: dict) -> pd.Series:
    """Return a cached rolling statistic, computing it on first use."""

    key = (column, window, stat)

    if key not in cache:

        r = df[column].rolling(window)

        if stat == "mean":
            cache[key] = r.mean()
        elif stat == "std":
            cache[key] = r.std()
        elif stat == "min":
            cache[key] = r.min()
        elif stat == "max":
            cache[key] = r.max()
        elif stat == "median":
            cache[key] = r.median()
        elif stat == "skew":
            cache[key] = r.skew()
        elif stat == "kurtosis":
            cache[key] = r.kurt()
        elif stat == "sum":
            cache[key] = r.sum()
        else:
            raise ValueError(f"Unknown rolling stat: {stat}")

    return cache[key]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a linear fit over each rolling window (trend direction/speed)."""

    def slope(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    return series.rolling(window).apply(slope, raw=True)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank (0-1) of the current value within its rolling window."""

    def pct(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        last = x[-1]
        return float((x <= last).sum()) / len(x)

    return series.rolling(window).apply(pct, raw=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Generate rolling window features: statistical, relative, and trend.

    Parameters
    ----------
    df : pd.DataFrame
    config : dict
        Per-column config. Supported keys:

        windows : list[int]
            Base windows used for the statistical block, e.g. [5, 10, 20, 50].

        mean, std, min, max, median, skew, kurtosis, sum : bool
            Statistical features, one per window in `windows`.

        relative : dict
            windows : list[int], optional (defaults to `windows`)
            dist_mean : bool   -> Close - rolling_mean   (distance from trend)
            ratio_mean : bool  -> Close / rolling_mean   (relative position)
            zscore : bool      -> (Close - mean) / std   (stretched-market detector)
            percentile : bool  -> percentile rank of current value in window

        trend : dict
            slope : list[int]
                Rolling linear-regression slope of the raw column per window.
            momentum : list[tuple[int, int]]
                (fast, slow) rolling-mean differences, e.g. [(5, 20)].
            crossover : list[tuple[int, int]]
                (fast, slow) sign of the rolling-mean difference (+1/-1/0),
                i.e. which mean is currently on top.
            std_ratio : list[tuple[int, int]]
                (short, long) std ratios, e.g. [(10, 50)] for volatility
                expansion/contraction.
            range : list[int]
                rolling_max - rolling_min per window.

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    for column, params in config.items():

        if column not in df.columns:
            raise ValueError(f"{column} not found.")

        cache: dict = {}
        col_lower = column.lower()
        windows = params.get("windows", [5, 10, 20])

        # ===================================================================
        # 1. Statistical Features
        # ===================================================================

        stat_flags = ["mean", "std", "min", "max", "median", "skew", "kurtosis", "sum"]

        for window in windows:
            for stat in stat_flags:
                if params.get(stat, False):
                    df[f"{col_lower}_{stat if stat != 'kurtosis' else 'kurt'}_{window}"] = (
                        _roll(df, column, window, stat, cache)
                    )

        # ===================================================================
        # 2. Relative Features
        # ===================================================================

        relative = params.get("relative", {})

        if relative:

            rel_windows = relative.get("windows", windows)

            for window in rel_windows:

                mean = _roll(df, column, window, "mean", cache)

                if relative.get("dist_mean", False):
                    df[f"{col_lower}_dist_mean_{window}"] = df[column] - mean

                if relative.get("ratio_mean", False):
                    df[f"{col_lower}_ratio_mean_{window}"] = df[column] / mean

                if relative.get("zscore", False):
                    std = _roll(df, column, window, "std", cache)
                    df[f"{col_lower}_zscore_{window}"] = (df[column] - mean) / std

                if relative.get("percentile", False):
                    df[f"{col_lower}_percentile_{window}"] = (
                        _rolling_percentile(df[column], window)
                    )

        # ===================================================================
        # 3. Trend Features
        # ===================================================================

        trend = params.get("trend", {})

        if trend:

            # --- Mean slope --------------------------------------------------
            for window in trend.get("slope", []):
                df[f"{col_lower}_slope_{window}"] = _rolling_slope(df[column], window)

            # --- Rolling momentum (fast mean - slow mean) --------------------
            for fast, slow in trend.get("momentum", []):
                fast_mean = _roll(df, column, fast, "mean", cache)
                slow_mean = _roll(df, column, slow, "mean", cache)
                df[f"{col_lower}_momentum_{fast}_{slow}"] = fast_mean - slow_mean

            # --- Mean crossover (sign of fast mean - slow mean) --------------
            for fast, slow in trend.get("crossover", []):
                fast_mean = _roll(df, column, fast, "mean", cache)
                slow_mean = _roll(df, column, slow, "mean", cache)
                df[f"{col_lower}_crossover_{fast}_{slow}"] = np.sign(fast_mean - slow_mean)

            # --- Std ratio / volatility expansion -----------------------------
            for short, long in trend.get("std_ratio", []):
                short_std = _roll(df, column, short, "std", cache)
                long_std = _roll(df, column, long, "std", cache)
                df[f"{col_lower}_std_ratio_{short}_{long}"] = short_std / long_std

            # --- Rolling range (max - min) -------------------------------------
            for window in trend.get("range", []):
                roll_max = _roll(df, column, window, "max", cache)
                roll_min = _roll(df, column, window, "min", cache)
                df[f"{col_lower}_range_{window}"] = roll_max - roll_min

    return df


# ---------------------------------------------------------------------------
# Example configuration using every feature family from the writeup
# ---------------------------------------------------------------------------

ROLLING_CONFIG = {

    "Close": {

        "windows": [5, 10, 20, 50],

        "mean": True,
        "std": True,
        "min": True,
        "max": True,
        "median": True,
        "skew": True,
        "kurtosis": True,

        "relative": {
            "windows": [20],
            "dist_mean": True,
            "ratio_mean": True,
            "zscore": True,
            "percentile": True,
        },

        "trend": {
            "slope": [20],
            "momentum": [(5, 20)],
            "crossover": [(5, 20)],
            "std_ratio": [(10, 50)],
            "range": [20],
        },
    },

    "Volume": {

        "windows": [5, 10, 20],

        "mean": True,
        "std": True,
        "max": True,
        "sum": True,

        "relative": {
            "windows": [20],
            "ratio_mean": True,
            "zscore": True,
        },
    },

    "ATR": {

        "windows": [10, 20],

        "mean": True,
        "std": True,

        "trend": {
            "std_ratio": [(10, 20)],
        },
    },
}


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
# df = add_rolling_features(df, config=ROLLING_CONFIG)
#
# Features created for "Close":
#   Statistical : close_mean_5/10/20/50, close_std_*, close_min_*, close_max_*,
#                 close_median_*, close_skew_*, close_kurt_*
#   Relative    : close_dist_mean_20, close_ratio_mean_20, close_zscore_20,
#                 close_percentile_20
#   Trend       : close_slope_20, close_momentum_5_20, close_crossover_5_20,
#                 close_std_ratio_10_50, close_range_20