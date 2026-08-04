"""
momentum_features.py

Momentum Feature Engineering

Author: RichGoons Project
"""

from __future__ import annotations

import pandas as pd

from ta.momentum import (
    RSIIndicator,
    ROCIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    AwesomeOscillatorIndicator,
    TSIIndicator,
)

from ta.trend import (
    MACD,
    CCIIndicator,
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


def add_momentum_features(df):

    _validate(df)

    df = df.copy()

    ####################################################
    # RSI
    ####################################################

    rsi = RSIIndicator(df["Close"])

    df["rsi"] = rsi.rsi()

    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

    df["rsi_above_50"] = (df["rsi"] > 50).astype(int)

    df["rsi_slope"] = df["rsi"].diff()

    df["rsi_acceleration"] = df["rsi_slope"].diff()

    ####################################################
    # MACD
    ####################################################

    macd = MACD(df["Close"])

    df["macd"] = macd.macd()

    df["macd_signal"] = macd.macd_signal()

    df["macd_hist"] = macd.macd_diff()

    df["macd_slope"] = df["macd"].diff()

    df["macd_hist_slope"] = df["macd_hist"].diff()

    df["macd_above_signal"] = (
        df["macd"] >
        df["macd_signal"]
    ).astype(int)

    df["macd_cross_up"] = (

        (df["macd"] > df["macd_signal"])

        &

        (
            df["macd"].shift(1)
            <=
            df["macd_signal"].shift(1)
        )

    ).astype(int)

    df["macd_cross_down"] = (

        (df["macd"] < df["macd_signal"])

        &

        (
            df["macd"].shift(1)
            >=
            df["macd_signal"].shift(1)
        )

    ).astype(int)

    ####################################################
    # STOCHASTIC
    ####################################################

    stoch = StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["stoch_k"] = stoch.stoch()

    df["stoch_d"] = stoch.stoch_signal()

    df["stoch_spread"] = (
        df["stoch_k"] -
        df["stoch_d"]
    )

    df["stoch_cross_up"] = (

        (df["stoch_k"] > df["stoch_d"])

        &

        (
            df["stoch_k"].shift(1)
            <=
            df["stoch_d"].shift(1)
        )

    ).astype(int)

    df["stoch_cross_down"] = (

        (df["stoch_k"] < df["stoch_d"])

        &

        (
            df["stoch_k"].shift(1)
            >=
            df["stoch_d"].shift(1)
        )

    ).astype(int)

    ####################################################
    # ROC
    ####################################################

    roc = ROCIndicator(df["Close"])

    df["roc"] = roc.roc()

    df["roc_slope"] = df["roc"].diff()

    ####################################################
    # CCI
    ####################################################

    cci = CCIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["cci"] = cci.cci()

    ####################################################
    # Williams %R
    ####################################################

    wr = WilliamsRIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["williams_r"] = wr.williams_r()

    ####################################################
    # Awesome Oscillator
    ####################################################

    ao = AwesomeOscillatorIndicator(
        high=df["High"],
        low=df["Low"]
    )

    df["awesome_oscillator"] = (
        ao.awesome_oscillator()
    )

    ####################################################
    # TSI
    ####################################################

    tsi = TSIIndicator(
        close=df["Close"]
    )

    df["tsi"] = tsi.tsi()

    df["tsi_slope"] = df["tsi"].diff()

    ####################################################
    # INDICATOR INTERACTION FEATURES
    ####################################################
    #
    # Tree models (XGBoost / LightGBM) split on one feature at a time,
    # so they can approximate interactions but need many splits to do
    # it well. Pre-multiplying related indicators hands the model the
    # interaction directly, which often shows up as high feature
    # importance on financial datasets.

    df["rsi_x_roc"] = df["rsi"] * df["roc"]

    df["macd_x_rsi"] = df["macd_hist"] * df["rsi"]

    df["stoch_x_rsi"] = df["stoch_k"] * df["rsi"]

    df["cci_x_roc"] = df["cci"] * df["roc"]

    ####################################################
    # COMPOSITE MOMENTUM FEATURES
    ####################################################

    df["momentum_score"] = (

        ((df["rsi"] - 50) / 50)

        +

        df["macd_hist"].fillna(0)

        +

        (df["roc"] / 100).fillna(0)

    )

    df["bullish_momentum"] = (

        (df["rsi"] > 50)

        &

        (df["macd"] > df["macd_signal"])

        &

        (df["roc"] > 0)

    ).astype(int)

    df["bearish_momentum"] = (

        (df["rsi"] < 50)

        &

        (df["macd"] < df["macd_signal"])

        &

        (df["roc"] < 0)

    ).astype(int)

    return df