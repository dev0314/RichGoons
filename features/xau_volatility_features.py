"""
volatility_features.py

Normalized Volatility Feature Engineering

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    KeltnerChannel,
)


REQUIRED_COLUMNS = [
    "High",
    "Low",
    "Close",
]


def _validate(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def add_xau_volatility_features(
    df: pd.DataFrame,
    atr_window: int = 14,
    bb_window: int = 20,
    bb_window_dev: int = 2,
    percentile_window: int = 100,
) -> pd.DataFrame:
    """
    Add normalized volatility features to an OHLC dataframe.

    Required columns:
        High, Low, Close

    Optional column:
        Open
        Required only if candle-structure features are added.

    Features generated:
        atr_pct
        ntr
        bb_width_pct
        bb_percent
        bb_kc_ratio
        volatility_expansion
        atr_ratio
        atr_zscore
        bb_zscore
        volatility_compression
        realized_vol_5
        realized_vol_10
        realized_vol_20
        realized_vol_50
        vol_ratio_5_20
        vol_ratio_10_50
        atr_percentile

    Returns:
        A copy of the input dataframe with volatility features added.
    """

    _validate(df)

    df = df.copy()

    # =========================================================
    # LOG RETURN
    # =========================================================

    log_return = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # =========================================================
    # TRUE RANGE
    # =========================================================

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] - previous_close
    ).abs()

    tr3 = (
        df["Low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # =========================================================
    # ATR
    # =========================================================

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=atr_window,
    )

    atr_14 = atr.average_true_range()

    # =========================================================
    # 1. ATR %
    # =========================================================

    df["atr_pct"] = (
        atr_14 / df["Close"]
    )

    # =========================================================
    # 2. NORMALIZED TRUE RANGE
    # =========================================================

    df["ntr"] = (
        true_range / df["Close"]
    )

    # =========================================================
    # BOLLINGER BANDS
    # =========================================================

    bb = BollingerBands(
        close=df["Close"],
        window=bb_window,
        window_dev=bb_window_dev,
    )

    bb_upper = bb.bollinger_hband()
    bb_middle = bb.bollinger_mavg()
    bb_lower = bb.bollinger_lband()

    bb_width = (
        bb_upper - bb_lower
    )

    # =========================================================
    # 3. BOLLINGER WIDTH %
    # =========================================================

    df["bb_width_pct"] = (
        bb_width / bb_middle
    )

    # =========================================================
    # 4. BOLLINGER %B
    # =========================================================

    df["bb_percent"] = (
        (df["Close"] - bb_lower)
        /
        bb_width
    )

    # =========================================================
    # KELTNER CHANNEL
    # =========================================================

    kc = KeltnerChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
    )

    kc_upper = kc.keltner_channel_hband()
    kc_lower = kc.keltner_channel_lband()

    kc_width = (
        kc_upper - kc_lower
    )

    # =========================================================
    # 5. BOLLINGER / KELTNER RATIO
    # =========================================================

    df["bb_kc_ratio"] = (
        bb_width / kc_width
    )

    # =========================================================
    # ROLLING VOLATILITY
    # =========================================================

    vol_20 = (
        log_return
        .rolling(20)
        .std()
    )

    # =========================================================
    # 6. VOLATILITY EXPANSION
    # =========================================================

    vol_20_mean = (
        vol_20
        .rolling(20)
        .mean()
    )

    df["volatility_expansion"] = (
        vol_20 / vol_20_mean
    )

    # =========================================================
    # 7. ATR RATIO
    # =========================================================

    atr_100_mean = (
        atr_14
        .rolling(100)
        .mean()
    )

    df["atr_ratio"] = (
        atr_14 / atr_100_mean
    )

    # =========================================================
    # 8. ATR Z-SCORE
    # =========================================================

    atr_mean_50 = (
        atr_14
        .rolling(50)
        .mean()
    )

    atr_std_50 = (
        atr_14
        .rolling(50)
        .std()
    )

    df["atr_zscore"] = (
        (atr_14 - atr_mean_50)
        / atr_std_50
    )

    # =========================================================
    # 9. BOLLINGER Z-SCORE
    # =========================================================

    rolling_price_std = (
        df["Close"]
        .rolling(bb_window)
        .std()
    )

    df["bb_zscore"] = (
        (df["Close"] - bb_middle)
        / rolling_price_std
    )

    # =========================================================
    # 10. VOLATILITY COMPRESSION
    # =========================================================

    vol_threshold = (
        vol_20
        .rolling(100)
        .quantile(0.20)
    )

    df["volatility_compression"] = (
        vol_20 < vol_threshold
    ).astype("int8")

    # =========================================================
    # 11-14. REALIZED VOLATILITY
    # =========================================================

    df["realized_vol_5"] = (
        log_return
        .rolling(5)
        .std()
    )

    df["realized_vol_10"] = (
        log_return
        .rolling(10)
        .std()
    )

    df["realized_vol_20"] = (
        log_return
        .rolling(20)
        .std()
    )

    df["realized_vol_50"] = (
        log_return
        .rolling(50)
        .std()
    )

    # =========================================================
    # 15. VOLATILITY RATIO 5 / 20
    # =========================================================

    df["vol_ratio_5_20"] = (
        df["realized_vol_5"]
        /
        df["realized_vol_20"]
    )

    # =========================================================
    # 16. VOLATILITY RATIO 10 / 50
    # =========================================================

    df["vol_ratio_10_50"] = (
        df["realized_vol_10"]
        /
        df["realized_vol_50"]
    )

    # =========================================================
    # 17. ATR PERCENTILE
    # =========================================================
    #
    # Current ATR(14) percentile within the trailing
    # 100 ATR(14) observations.
    #
    # This intentionally follows the original working
    # implementation rather than using rolling().rank(),
    # because the lambda explicitly extracts the rank
    # of the current/latest ATR in each window.

    df["atr_percentile"] = (
        atr_14
        .rolling(percentile_window)
        .apply(
            lambda x: (
                pd.Series(x)
                .rank(pct=True)
                .iloc[-1]
            ),
            raw=False,
        )
    )

    # =========================================================
    # CLEANUP
    # =========================================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return df


XAU_VOLATILITY_FEATURES = [
    "atr_pct",
    "ntr",
    "bb_width_pct",
    "bb_percent",
    "bb_kc_ratio",
    "volatility_expansion",
    "atr_ratio",
    "atr_zscore",
    "bb_zscore",
    "volatility_compression",
    "realized_vol_5",
    "realized_vol_10",
    "realized_vol_20",
    "realized_vol_50",
    "vol_ratio_5_20",
    "vol_ratio_10_50",
    "atr_percentile",
]
