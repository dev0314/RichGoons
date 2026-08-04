"""
volatility_features.py

Volatility Feature Engineering

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel
)

REQUIRED_COLUMNS = [
    "High",
    "Low",
    "Close"
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


def add_volatility_features(
    df: pd.DataFrame,
    rolling_window: int = 20,
    realized_vol_windows: list = [5, 10, 20, 50],
):

    _validate(df)

    df = df.copy()

    ####################################################
    # TRUE RANGE
    ####################################################

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    df["true_range"] = np.maximum.reduce(
        [tr1, tr2, tr3]
    )

    ####################################################
    # ATR
    ####################################################

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["atr"] = (
        atr.average_true_range()
    )

    df["atr_pct"] = (
        df["atr"] /
        df["Close"]
    )

    df["atr_slope"] = (
        df["atr"].diff()
    )

    ####################################################
    # HISTORICAL VOLATILITY
    ####################################################

    log_return = np.log(
        df["Close"] /
        df["Close"].shift(1)
    )

    df["historical_volatility"] = (

        log_return

        .rolling(rolling_window)

        .std()

        *

        np.sqrt(365 * 24 * 12)

    )

    ####################################################
    # BOLLINGER BANDS
    ####################################################

    bb = BollingerBands(
        close=df["Close"]
    )

    df["bb_upper"] = (
        bb.bollinger_hband()
    )

    df["bb_middle"] = (
        bb.bollinger_mavg()
    )

    df["bb_lower"] = (
        bb.bollinger_lband()
    )

    df["bb_width"] = (

        df["bb_upper"]

        -

        df["bb_lower"]

    )

    df["bb_width_pct"] = (

        df["bb_width"]

        /

        df["bb_middle"]

    )

    df["bb_percent"] = (

        (

            df["Close"]

            -

            df["bb_lower"]

        )

        /

        (

            df["bb_upper"]

            -

            df["bb_lower"]

        )

    )

    ####################################################
    # DONCHIAN CHANNEL
    ####################################################

    dc = DonchianChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["dc_upper"] = (
        dc.donchian_channel_hband()
    )

    df["dc_lower"] = (
        dc.donchian_channel_lband()
    )

    df["dc_middle"] = (
        dc.donchian_channel_mband()
    )

    ####################################################
    # KELTNER CHANNEL
    ####################################################

    kc = KeltnerChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["kc_upper"] = (
        kc.keltner_channel_hband()
    )

    df["kc_lower"] = (
        kc.keltner_channel_lband()
    )

    df["kc_middle"] = (
        kc.keltner_channel_mband()
    )

    ####################################################
    # ROLLING VOLATILITY
    ####################################################

    df["rolling_std"] = (

        df["Close"]

        .rolling(rolling_window)

        .std()

    )

    df["rolling_range"] = (

        df["High"]

        .rolling(rolling_window)

        .max()

        -

        df["Low"]

        .rolling(rolling_window)

        .min()

    )

    ####################################################
    # VOLATILITY EXPANSION
    ####################################################

    df["volatility_expansion"] = (

        df["rolling_std"]

        /

        df["rolling_std"]

        .rolling(20)

        .mean()

    )

    ####################################################
    # ATR Z SCORE
    ####################################################

    atr_mean = (

        df["atr"]

        .rolling(50)

        .mean()

    )

    atr_std = (

        df["atr"]

        .rolling(50)

        .std()

    )

    df["atr_zscore"] = (

        (

            df["atr"]

            -

            atr_mean

        )

        /

        atr_std

    )

    ####################################################
    # VOLATILITY REGIME
    ####################################################

    df["high_volatility"] = (

        df["atr_zscore"] > 1

    ).astype(int)

    df["low_volatility"] = (

        df["atr_zscore"] < -1

    ).astype(int)

    ####################################################
    # BB SQUEEZE
    ####################################################

    bb_width_mean = (

        df["bb_width"]

        .rolling(50)

        .mean()

    )

    df["bb_squeeze"] = (

        df["bb_width"]

        <

        bb_width_mean * 0.75

    ).astype(int)

    ####################################################
    # BB BREAKOUT
    ####################################################

    df["bb_breakout_up"] = (

        df["Close"]

        >

        df["bb_upper"]

    ).astype(int)

    df["bb_breakout_down"] = (

        df["Close"]

        <

        df["bb_lower"]

    ).astype(int)

    ####################################################
    # NORMALIZED TRUE RANGE
    ####################################################
    # True range as a fraction of price, so it's comparable
    # across instruments/price levels.

    df["ntr"] = (

        df["true_range"]

        /

        df["Close"]

    )

    ####################################################
    # ATR RATIO (relative to long-term average)
    ####################################################

    df["atr_ratio"] = (

        df["atr"]

        /

        df["atr"].rolling(100).mean()

    )

    ####################################################
    # BOLLINGER BAND Z-SCORE
    ####################################################
    # Where price sits relative to the middle band, scaled
    # by rolling std — similar spirit to atr_zscore but for
    # price location rather than volatility level.

    df["bb_zscore"] = (

        (

            df["Close"]

            -

            df["bb_middle"]

        )

        /

        df["rolling_std"]

    )

    ####################################################
    # CHANNEL WIDTH COMPARISON (Bollinger vs Keltner)
    ####################################################
    # A classic "squeeze" definition: Bollinger Bands
    # narrower than Keltner Channels signals compressed
    # volatility often preceding a breakout.

    df["bb_kc_ratio"] = (

        df["bb_width"]

        /

        (

            df["kc_upper"]

            -

            df["kc_lower"]

        )

    )

    ####################################################
    # VOLATILITY ACCELERATION
    ####################################################
    # Second derivative of ATR — is volatility itself
    # speeding up or slowing down.

    df["atr_acceleration"] = (

        df["atr_slope"].diff()

    )

    ####################################################
    # VOLATILITY COMPRESSION
    ####################################################
    # Flags periods where current volatility sits in the
    # bottom 20% of its trailing 100-period distribution —
    # a classic pre-breakout "coiling" signal.

    df["volatility_compression"] = (

        df["rolling_std"]

        <

        df["rolling_std"]

        .rolling(100)

        .quantile(0.2)

    ).astype(int)

    ####################################################
    # REALIZED VOLATILITY (multiple horizons)
    ####################################################
    # log_return already computed above for historical_volatility

    for window in realized_vol_windows:

        df[f"realized_vol_{window}"] = (

            log_return

            .rolling(window)

            .std()

            *

            np.sqrt(window)

        )

    ####################################################
    # REALIZED VOLATILITY RATIOS
    ####################################################
    # Short-vs-long realized vol ratios — is the market
    # transitioning into a new volatility regime.

    if 5 in realized_vol_windows and 20 in realized_vol_windows:

        df["vol_ratio_5_20"] = (

            df["realized_vol_5"]

            /

            df["realized_vol_20"]

        )

    if 10 in realized_vol_windows and 50 in realized_vol_windows:

        df["vol_ratio_10_50"] = (

            df["realized_vol_10"]

            /

            df["realized_vol_50"]

        )

    ####################################################
    # ATR PERCENTILE
    ####################################################
    # NOTE: Rolling.rank() does NOT compute the percentile
    # rank of the *latest* value within each window the way
    # you'd want here — it ranks every point in the window
    # against the others, which is not the same thing.
    # Using .apply() with pandas rank(pct=True) on each
    # window and taking the last element gives the correct
    # "where does today's ATR sit vs. the last 100 readings"
    # answer.

    df["atr_percentile"] = (

        df["atr"]

        .rolling(100)

        .apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

    )

    return df