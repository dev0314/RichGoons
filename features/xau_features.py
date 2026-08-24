"""
xau_features.py

Normalized Price, Candle Structure, and Volatility
Feature Engineering for XAUUSD.

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


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
]


# =========================================================
# VALIDATION
# =========================================================

def _validate(
    df: pd.DataFrame
) -> None:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_xau_features(
    df: pd.DataFrame,
    atr_window: int = 14,
    bb_window: int = 20,
    bb_window_dev: int = 2,
    percentile_window: int = 100,
) -> pd.DataFrame:

    """
    Add normalized price, candle, and volatility
    features to an XAUUSD OHLC dataframe.

    Required columns:
        Open
        High
        Low
        Close

    Returns:
        Copy of dataframe with engineered features.
    """

    _validate(df)

    df = df.copy()

    # =====================================================
    # PREVIOUS CLOSE
    # =====================================================

    previous_close = (
        df["Close"].shift(1)
    )

    # =====================================================
    # PRICE FEATURES
    # =====================================================

    # -----------------------------------------------------
    # LOG RETURN
    # -----------------------------------------------------

    df["log_return"] = np.log(
        df["Close"]
        / previous_close
    )

    # -----------------------------------------------------
    # MULTI-HORIZON RETURNS
    # -----------------------------------------------------

    df["return_3"] = (
        df["Close"]
        / df["Close"].shift(3)
        - 1
    )

    df["return_6"] = (
        df["Close"]
        / df["Close"].shift(6)
        - 1
    )

    df["return_12"] = (
        df["Close"]
        / df["Close"].shift(12)
        - 1
    )

    df["return_24"] = (
        df["Close"]
        / df["Close"].shift(24)
        - 1
    )

    # -----------------------------------------------------
    # NORMALIZED OHLC
    # -----------------------------------------------------

    df["open_pct"] = (
        df["Open"]
        / previous_close
        - 1
    )

    df["high_pct"] = (
        df["High"]
        / previous_close
        - 1
    )

    df["low_pct"] = (
        df["Low"]
        / previous_close
        - 1
    )

    df["close_pct"] = (
        df["Close"]
        / previous_close
        - 1
    )

    # =====================================================
    # CANDLE FEATURES
    # =====================================================

    # -----------------------------------------------------
    # BODY %
    # -----------------------------------------------------

    df["body_pct"] = (
        (
            df["Close"]
            - df["Open"]
        )
        / previous_close
    )

    # -----------------------------------------------------
    # RANGE %
    # -----------------------------------------------------

    candle_range = (
        df["High"]
        - df["Low"]
    )

    df["range_pct"] = (
        candle_range
        / df["Close"]
    )

    # -----------------------------------------------------
    # CANDLE TOP / BOTTOM
    # -----------------------------------------------------

    candle_top = df[
        ["Open", "Close"]
    ].max(axis=1)

    candle_bottom = df[
        ["Open", "Close"]
    ].min(axis=1)

    # -----------------------------------------------------
    # UPPER WICK %
    # -----------------------------------------------------

    df["upper_wick_pct"] = (
        (
            df["High"]
            - candle_top
        )
        / df["Close"]
    )

    # -----------------------------------------------------
    # LOWER WICK %
    # -----------------------------------------------------

    df["lower_wick_pct"] = (
        (
            candle_bottom
            - df["Low"]
        )
        / df["Close"]
    )

    # -----------------------------------------------------
    # BODY TO RANGE
    # -----------------------------------------------------

    df["body_to_range"] = (
        (
            df["Close"]
            - df["Open"]
        ).abs()
        / candle_range
    )

    # =====================================================
    # LOG RETURN FOR VOLATILITY
    # =====================================================

    log_return = df["log_return"]

    # =====================================================
    # TRUE RANGE
    # =====================================================

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = (
        df["High"]
        - previous_close
    ).abs()

    tr3 = (
        df["Low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    # =====================================================
    # ATR
    # =====================================================

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=atr_window,
    )

    atr_14 = (
        atr.average_true_range()
    )

    # =====================================================
    # 1. ATR %
    # =====================================================

    df["atr_pct"] = (
        atr_14
        / df["Close"]
    )

    # =====================================================
    # 2. NORMALIZED TRUE RANGE
    # =====================================================

    df["ntr"] = (
        true_range
        / df["Close"]
    )

    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    bb = BollingerBands(
        close=df["Close"],
        window=bb_window,
        window_dev=bb_window_dev,
    )

    bb_upper = (
        bb.bollinger_hband()
    )

    bb_middle = (
        bb.bollinger_mavg()
    )

    bb_lower = (
        bb.bollinger_lband()
    )

    bb_width = (
        bb_upper
        - bb_lower
    )

    # =====================================================
    # 3. BOLLINGER WIDTH %
    # =====================================================

    df["bb_width_pct"] = (
        bb_width
        / bb_middle
    )

    # =====================================================
    # 4. BOLLINGER %B
    # =====================================================

    df["bb_percent"] = (
        (
            df["Close"]
            - bb_lower
        )
        / bb_width
    )

    # =====================================================
    # KELTNER CHANNEL
    # =====================================================

    kc = KeltnerChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
    )

    kc_upper = (
        kc.keltner_channel_hband()
    )

    kc_lower = (
        kc.keltner_channel_lband()
    )

    kc_width = (
        kc_upper
        - kc_lower
    )

    # =====================================================
    # 5. BB / KC RATIO
    # =====================================================

    df["bb_kc_ratio"] = (
        bb_width
        / kc_width
    )

    # =====================================================
    # ROLLING VOLATILITY
    # =====================================================

    vol_20 = (
        log_return
        .rolling(20)
        .std()
    )

    # =====================================================
    # 6. VOLATILITY EXPANSION
    # =====================================================

    vol_20_mean = (
        vol_20
        .rolling(20)
        .mean()
    )

    df["volatility_expansion"] = (
        vol_20
        / vol_20_mean
    )

    # =====================================================
    # 7. ATR RATIO
    # =====================================================

    atr_100_mean = (
        atr_14
        .rolling(100)
        .mean()
    )

    df["atr_ratio"] = (
        atr_14
        / atr_100_mean
    )

    # =====================================================
    # 8. ATR Z-SCORE
    # =====================================================

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
        (
            atr_14
            - atr_mean_50
        )
        / atr_std_50
    )

    # =====================================================
    # 9. BOLLINGER Z-SCORE
    # =====================================================

    rolling_price_std = (
        df["Close"]
        .rolling(bb_window)
        .std()
    )

    df["bb_zscore"] = (
        (
            df["Close"]
            - bb_middle
        )
        / rolling_price_std
    )

    # =====================================================
    # 10. VOLATILITY COMPRESSION
    # =====================================================

    vol_threshold = (
        vol_20
        .rolling(100)
        .quantile(0.20)
    )

    df["volatility_compression"] = (
        vol_20
        < vol_threshold
    ).astype("int8")

    # =====================================================
    # 11. REALIZED VOLATILITY 5
    # =====================================================

    df["realized_vol_5"] = (
        log_return
        .rolling(5)
        .std()
    )

    # =====================================================
    # 12. REALIZED VOLATILITY 10
    # =====================================================

    df["realized_vol_10"] = (
        log_return
        .rolling(10)
        .std()
    )

    # =====================================================
    # 13. REALIZED VOLATILITY 20
    # =====================================================

    df["realized_vol_20"] = (
        log_return
        .rolling(20)
        .std()
    )

    # =====================================================
    # 14. REALIZED VOLATILITY 50
    # =====================================================

    df["realized_vol_50"] = (
        log_return
        .rolling(50)
        .std()
    )

    # =====================================================
    # 15. VOLATILITY RATIO 5 / 20
    # =====================================================

    df["vol_ratio_5_20"] = (
        df["realized_vol_5"]
        / df["realized_vol_20"]
    )

    # =====================================================
    # 16. VOLATILITY RATIO 10 / 50
    # =====================================================

    df["vol_ratio_10_50"] = (
        df["realized_vol_10"]
        / df["realized_vol_50"]
    )

    # =====================================================
    # 17. ATR PERCENTILE
    # =====================================================

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

    # =====================================================
    # CLEANUP
    # =====================================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return df

# =========================================================
# PRICE FEATURES
# =========================================================

XAU_PRICE_FEATURES = [
    "log_return",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "open_pct",
    "high_pct",
    "low_pct",
    "close_pct",
]


# =========================================================
# CANDLE FEATURES
# =========================================================

XAU_CANDLE_FEATURES = [
    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "body_to_range",
]


# =========================================================
# VOLATILITY FEATURES
# =========================================================

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


# =========================================================
# ALL XAU FEATURES
# =========================================================

XAU_FEATURES = (
    XAU_PRICE_FEATURES
    + XAU_CANDLE_FEATURES
    + XAU_VOLATILITY_FEATURES
)