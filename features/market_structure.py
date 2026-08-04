"""
market_structure.py

Market Structure Features

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "High",
    "Low",
    "Close",
    "Volume"
]


def _validate(df):

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def add_market_structure_features(
    df: pd.DataFrame,
    lookback=20,
    swing_order: int = 3,
    fvg_min_gap: float = 0.0,
    liquidity_tolerance: float = 0.001,
    volume_spike_mult: float = 1.5,
    trend_window: int = 10,
):
    """
    Adds market structure features, including swing-based / smart-money
    style features: swing highs & lows, HH/HL/LH/LL classification,
    Break of Structure (BOS), Change of Character (CHoCH), Fair Value
    Gaps (FVG), liquidity pools, volume-confirmed breakouts, trend
    quality, and a composite structure score.

    ################################################################
    # IMPORTANT - LOOKAHEAD BIAS
    ################################################################
    Swing highs/lows are confirmed using a centered window, i.e. they
    look `swing_order` bars into the FUTURE to decide whether bar t is
    a swing point. This is standard for descriptive/backtest structure
    analysis, but it means swing_high, swing_low, and everything derived
    from them (is_hh, is_hl, bos_*, choch_*, structure_shift,
    liquidity_pool_*, structure_score) are NOT causal.

    If you plan to feed these into an XGBoost/LightGBM model that
    predicts forward price movement, you must shift these columns by
    at least `swing_order` bars (df[cols].shift(swing_order)) before
    using them as features, or you will leak future information into
    the training set.
    ################################################################
    """

    _validate(df)

    df = df.copy()

    ####################################################
    # Rolling High / Low
    ####################################################

    df["rolling_high"] = (
        df["High"]
        .rolling(lookback)
        .max()
    )

    df["rolling_low"] = (
        df["Low"]
        .rolling(lookback)
        .min()
    )

    ####################################################
    # Distance to High / Low
    ####################################################

    df["dist_high"] = (
        df["rolling_high"]
        -
        df["Close"]
    )

    df["dist_low"] = (
        df["Close"]
        -
        df["rolling_low"]
    )

    ####################################################
    # Breakout
    ####################################################

    df["breakout_up"] = (

        df["Close"]

        >

        df["rolling_high"].shift(1)

    ).astype(int)

    df["breakout_down"] = (

        df["Close"]

        <

        df["rolling_low"].shift(1)

    ).astype(int)

    ####################################################
    # Higher High
    ####################################################

    df["higher_high"] = (

        df["High"]

        >

        df["High"].shift(1)

    ).astype(int)

    ####################################################
    # Lower Low
    ####################################################

    df["lower_low"] = (

        df["Low"]

        <

        df["Low"].shift(1)

    ).astype(int)

    ####################################################
    # Higher Close
    ####################################################

    df["higher_close"] = (

        df["Close"]

        >

        df["Close"].shift(1)

    ).astype(int)

    ####################################################
    # Lower Close
    ####################################################

    df["lower_close"] = (

        df["Close"]

        <

        df["Close"].shift(1)

    ).astype(int)

    ####################################################
    # Consecutive Higher Highs
    ####################################################

    df["hh_count"] = (

        df["higher_high"]

        .rolling(5)

        .sum()

    )

    ####################################################
    # Consecutive Lower Lows
    ####################################################

    df["ll_count"] = (

        df["lower_low"]

        .rolling(5)

        .sum()

    )

    ####################################################
    # Price Compression
    ####################################################

    df["compression"] = (

        (
            df["High"]

            -

            df["Low"]

        )

        <

        (

            df["High"]

            -

            df["Low"]

        )

        .rolling(20)

        .mean()

        *0.5

    ).astype(int)

    ####################################################
    # Range Expansion
    ####################################################

    df["range_expansion"] = (

        (
            df["High"]

            -

            df["Low"]

        )

        >

        (

            df["High"]

            -

            df["Low"]

        )

        .rolling(20)

        .mean()

        *2

    ).astype(int)

    ####################################################
    # Liquidity Sweep High
    ####################################################

    df["liquidity_sweep_high"] = (

        (df["High"]

         >

         df["rolling_high"].shift(1))

        &

        (df["Close"]

         <

         df["rolling_high"].shift(1))

    ).astype(int)

    ####################################################
    # Liquidity Sweep Low
    ####################################################

    df["liquidity_sweep_low"] = (

        (df["Low"]

         <

         df["rolling_low"].shift(1))

        &

        (df["Close"]

         >

         df["rolling_low"].shift(1))

    ).astype(int)

    ####################################################
    # Breakout Strength
    ####################################################

    df["breakout_strength"] = (

        (df["Close"]

         -

         df["rolling_high"].shift(1))

        /

        df["Close"]

    )

    ####################################################
    # Pullback
    ####################################################

    # NOTE: requires "ema_20" / "ema_50" columns to already exist on df
    # (e.g. from a separate trend/EMA feature module). This function does
    # not compute them itself.

    df["pullback"] = (

        (df["Close"]

         <

         df["ema_20"])

        &

        (df["ema_20"]

         >

         df["ema_50"])

    ).astype(int)

    ####################################################
    # Swing High / Swing Low Detection
    ####################################################

    window = 2 * swing_order + 1

    roll_max = (
        df["High"]
        .rolling(window, center=True)
        .max()
    )

    roll_min = (
        df["Low"]
        .rolling(window, center=True)
        .min()
    )

    df["swing_high"] = (df["High"] == roll_max).astype(int)
    df["swing_low"] = (df["Low"] == roll_min).astype(int)

    ####################################################
    # Sparse Swing Point Prices (NaN where no swing)
    ####################################################

    df["swing_high_price"] = np.where(
        df["swing_high"] == 1, df["High"], np.nan
    )

    df["swing_low_price"] = np.where(
        df["swing_low"] == 1, df["Low"], np.nan
    )

    ####################################################
    # HH / LH classification at each confirmed swing high
    ####################################################

    sh = df["swing_high_price"].dropna()
    prev_sh = sh.shift(1)

    df["is_hh"] = 0
    df["is_lh"] = 0
    df.loc[sh.index, "is_hh"] = (sh > prev_sh).astype(int)
    df.loc[sh.index, "is_lh"] = (sh < prev_sh).astype(int)

    ####################################################
    # HL / LL classification at each confirmed swing low
    ####################################################

    sl = df["swing_low_price"].dropna()
    prev_sl = sl.shift(1)

    df["is_hl"] = 0
    df["is_ll"] = 0
    df.loc[sl.index, "is_hl"] = (sl > prev_sl).astype(int)
    df.loc[sl.index, "is_ll"] = (sl < prev_sl).astype(int)

    ####################################################
    # Persist last structure label between swing points
    ####################################################

    high_label = pd.Series(
        np.where(
            df["is_hh"] == 1, "HH",
            np.where(df["is_lh"] == 1, "LH", None)
        ),
        index=df.index,
        dtype=object,
    )

    low_label = pd.Series(
        np.where(
            df["is_hl"] == 1, "HL",
            np.where(df["is_ll"] == 1, "LL", None)
        ),
        index=df.index,
        dtype=object,
    )

    df["last_high_label"] = high_label.ffill()
    df["last_low_label"] = low_label.ffill()

    ####################################################
    # Trend State: bull structure (HH+HL) vs bear (LH+LL)
    ####################################################

    df["bull_structure"] = (
        (df["last_high_label"] == "HH")
        & (df["last_low_label"] == "HL")
    ).astype(int)

    df["bear_structure"] = (
        (df["last_high_label"] == "LH")
        & (df["last_low_label"] == "LL")
    ).astype(int)

    ####################################################
    # Break of Structure (BOS) - trend continuation
    ####################################################

    last_confirmed_high = df["swing_high_price"].ffill()
    last_confirmed_low = df["swing_low_price"].ffill()

    df["bos_up"] = (
        (df["Close"] > last_confirmed_high.shift(1))
        & (df["bull_structure"].shift(1) == 1)
    ).astype(int)

    df["bos_down"] = (
        (df["Close"] < last_confirmed_low.shift(1))
        & (df["bear_structure"].shift(1) == 1)
    ).astype(int)

    ####################################################
    # Change of Character (CHoCH) - trend reversal
    ####################################################

    df["choch_bear"] = (
        (df["bull_structure"].shift(1) == 1)
        & (df["Close"] < last_confirmed_low.shift(1))
    ).astype(int)

    df["choch_bull"] = (
        (df["bear_structure"].shift(1) == 1)
        & (df["Close"] > last_confirmed_high.shift(1))
    ).astype(int)

    ####################################################
    # Market Structure Shift (either direction)
    ####################################################

    df["structure_shift"] = (
        (df["choch_bull"] == 1) | (df["choch_bear"] == 1)
    ).astype(int)

    ####################################################
    # Fair Value Gap (FVG)
    ####################################################

    df["fvg_bullish"] = (
        df["Low"] > df["High"].shift(2) + fvg_min_gap
    ).astype(int)

    df["fvg_bearish"] = (
        df["High"] < df["Low"].shift(2) - fvg_min_gap
    ).astype(int)

    df["fvg_size"] = np.where(
        df["fvg_bullish"] == 1,
        df["Low"] - df["High"].shift(2),
        np.where(
            df["fvg_bearish"] == 1,
            df["Low"].shift(2) - df["High"],
            0.0,
        ),
    )

    ####################################################
    # Liquidity Pool Detection
    ####################################################
    # Flags a swing high/low that sits within `liquidity_tolerance`
    # (relative pct) of an EARLIER swing high/low -> a repeated level,
    # i.e. a likely resting stop-loss / liquidity cluster.

    df["liquidity_pool_high"] = 0
    df["liquidity_pool_low"] = 0

    sh_prices = df["swing_high_price"].dropna()
    liquidity_high_idx = []

    for i in range(1, len(sh_prices)):
        cur = sh_prices.iloc[i]
        prior = sh_prices.iloc[:i]
        if ((prior - cur).abs() / cur < liquidity_tolerance).any():
            liquidity_high_idx.append(sh_prices.index[i])

    if liquidity_high_idx:
        df.loc[liquidity_high_idx, "liquidity_pool_high"] = 1

    sl_prices = df["swing_low_price"].dropna()
    liquidity_low_idx = []

    for i in range(1, len(sl_prices)):
        cur = sl_prices.iloc[i]
        prior = sl_prices.iloc[:i]
        if ((prior - cur).abs() / cur < liquidity_tolerance).any():
            liquidity_low_idx.append(sl_prices.index[i])

    if liquidity_low_idx:
        df.loc[liquidity_low_idx, "liquidity_pool_low"] = 1

    ####################################################
    # Volume-Confirmed Breakout
    ####################################################

    vol_ma = df["Volume"].rolling(20).mean()
    df["volume_spike"] = (df["Volume"] > vol_ma * volume_spike_mult).astype(int)

    if "breakout_up" in df.columns and "breakout_down" in df.columns:
        breakout_up = df["breakout_up"]
        breakout_down = df["breakout_down"]
    else:
        rolling_high = df["High"].rolling(20).max()
        rolling_low = df["Low"].rolling(20).min()
        breakout_up = (df["Close"] > rolling_high.shift(1)).astype(int)
        breakout_down = (df["Close"] < rolling_low.shift(1)).astype(int)

    df["breakout_up_confirmed"] = (
        (breakout_up == 1) & (df["volume_spike"] == 1)
    ).astype(int)

    df["breakout_down_confirmed"] = (
        (breakout_down == 1) & (df["volume_spike"] == 1)
    ).astype(int)

    ####################################################
    # Trend Quality Score (0-10)
    ####################################################
    # Rewards a rolling window of confirmed swings that keep making
    # HH/HL (uptrend) and penalizes ones making LH/LL (downtrend).

    hh_hl_streak = ((df["is_hh"] == 1) | (df["is_hl"] == 1)).astype(int)
    lh_ll_streak = ((df["is_lh"] == 1) | (df["is_ll"] == 1)).astype(int)

    raw_score = (
        hh_hl_streak.rolling(trend_window).sum()
        - lh_ll_streak.rolling(trend_window).sum()
    ).clip(-trend_window, trend_window)

    df["trend_quality"] = (
        (raw_score + trend_window) / (2 * trend_window) * 10
    ).round(2)

    ####################################################
    # Composite Structure Score
    ####################################################
    # Weighted roll-up of the structural signals above into a single
    # feature. Weights are a reasonable starting point - tune them
    # (or let a model learn on the components directly) for your use
    # case.

    compression_col = (
        df["compression"] if "compression" in df.columns else 0
    )

    df["structure_score"] = (
        df["bull_structure"] * 2
        - df["bear_structure"] * 2
        + df["bos_up"] * 2
        - df["bos_down"] * 2
        + df["choch_bull"] * 3
        - df["choch_bear"] * 3
        + df["breakout_up_confirmed"] * 2
        - df["breakout_down_confirmed"] * 2
        - compression_col * 1
        - df["liquidity_pool_high"] * 1
        + df["liquidity_pool_low"] * 1
    )

    return df