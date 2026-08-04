"""
volume_features.py

Volume Feature Engineering

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.volume import (
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator,
    MFIIndicator,
    ForceIndexIndicator,
)

REQUIRED_COLUMNS = [
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Volume",
    "Taker Buy Quote Volume"
]


def _validate(df):

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns {missing}"
        )


def add_volume_features(df):

    _validate(df)

    df = df.copy()

    ####################################################
    # BASIC BUY / SELL VOLUME
    ####################################################

    df["taker_sell_base_volume"] = (
        df["Volume"]
        -
        df["Taker Buy Base Volume"]
    )

    df["taker_sell_quote_volume"] = (
        df["Quote Asset Volume"]
        -
        df["Taker Buy Quote Volume"]
    )

    ####################################################
    # BUY RATIO
    ####################################################

    df["buy_volume_ratio"] = (
        df["Taker Buy Base Volume"]
        /
        df["Volume"]
    )

    df["sell_volume_ratio"] = (
        df["taker_sell_base_volume"]
        /
        df["Volume"]
    )

    ####################################################
    # BUY PRESSURE
    ####################################################

    df["buy_pressure"] = (

        df["Taker Buy Base Volume"]

        -

        df["taker_sell_base_volume"]

    )

    ####################################################
    # TRADE SIZE
    ####################################################

    df["avg_trade_size"] = (

        df["Volume"]

        /

        df["Number of Trades"]

    )

    df["avg_trade_value"] = (

        df["Quote Asset Volume"]

        /

        df["Number of Trades"]

    )

    ####################################################
    # ROLLING VOLUME
    ####################################################

    for window in [5,10,20,50]:

        rolling_mean = (
            df["Volume"]
            .rolling(window)
            .mean()
        )

        rolling_std = (
            df["Volume"]
            .rolling(window)
            .std()
        )

        df[f"volume_mean_{window}"] = rolling_mean

        df[f"volume_std_{window}"] = rolling_std

        df[f"volume_ratio_{window}"] = (
            df["Volume"] /
            rolling_mean
        )

        df[f"volume_zscore_{window}"] = (

            (

                df["Volume"]

                -

                rolling_mean

            )

            /

            rolling_std

        )

    ####################################################
    # VOLUME SLOPE
    ####################################################

    df["volume_slope"] = (
        df["Volume"].diff()
    )

    df["volume_acceleration"] = (
        df["volume_slope"].diff()
    )

    ####################################################
    # BUY PRESSURE SLOPE
    ####################################################

    df["buy_pressure_slope"] = (
        df["buy_pressure"].diff()
    )

    ####################################################
    # OBV
    ####################################################

    obv = OnBalanceVolumeIndicator(
        close=df["Close"],
        volume=df["Volume"]
    )

    df["obv"] = obv.on_balance_volume()

    ####################################################
    # FORCE INDEX
    ####################################################

    fi = ForceIndexIndicator(
        close=df["Close"],
        volume=df["Volume"]
    )

    df["force_index"] = fi.force_index()

    ####################################################
    # MONEY FLOW INDEX
    ####################################################

    mfi = MFIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )

    df["mfi"] = mfi.money_flow_index()

    ####################################################
    # CHAIKIN MONEY FLOW
    ####################################################

    cmf = ChaikinMoneyFlowIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )

    df["cmf"] = cmf.chaikin_money_flow()

    ####################################################
    # VOLUME SPIKE
    ####################################################

    df["volume_spike"] = (

        df["volume_ratio_20"]

        > 2

    ).astype(int)

    ####################################################
    # BUY IMBALANCE
    ####################################################

    df["buy_imbalance"] = (

        df["buy_volume_ratio"]

        > 0.60

    ).astype(int)

    ####################################################
    # SELL IMBALANCE
    ####################################################

    df["sell_imbalance"] = (

        df["sell_volume_ratio"]

        > 0.60

    ).astype(int)

    ####################################################
    # RELATIVE TRADE SIZE
    ####################################################
    # Is the average trade right now bigger or smaller than
    # the recent norm — spikes can flag institutional/whale
    # activity vs. a crowd of small retail orders.

    df["relative_trade_size"] = (

        df["avg_trade_size"]

        /

        df["avg_trade_size"]

        .rolling(50)

        .mean()

    )

    ####################################################
    # QUOTE / BASE RATIO
    ####################################################
    # Effectively the volume-weighted average price for the
    # bar; useful as a sanity check and as its own feature
    # since it reacts to intra-bar price distribution.

    df["quote_base_ratio"] = (

        df["Quote Asset Volume"]

        /

        df["Volume"]

    )

    ####################################################
    # BUYER DOMINANCE
    ####################################################

    df["buyer_dominance"] = (

        df["buy_volume_ratio"]

        -

        df["sell_volume_ratio"]

    )

    ####################################################
    # ROLLING BUYER DOMINANCE
    ####################################################
    # Smooths out single-bar noise to show whether buyers
    # have had the upper hand over a sustained stretch.

    df["buyer_dominance_ma"] = (

        df["buyer_dominance"]

        .rolling(20)

        .mean()

    )

    ####################################################
    # VOLUME x PRICE CHANGE
    ####################################################
    # Large volume on a large price move = conviction.
    # Large volume on a flat move = potential absorption /
    # distribution.

    df["volume_price_strength"] = (

        df["Volume"]

        *

        df["Close"]

        .pct_change()

    )

    ####################################################
    # VOLUME x ATR (cross-module interaction)
    ####################################################
    # Only added if volatility_features.py has already run
    # on this dataframe and contributed an "atr" column.

    if "atr" in df.columns:

        df["volume_atr"] = (

            df["Volume"]

            *

            df["atr"]

        )

    ####################################################
    # VOLUME x RSI (cross-module interaction)
    ####################################################
    # Only added if momentum_features.py has already run
    # on this dataframe and contributed an "rsi" column.

    if "rsi" in df.columns:

        df["volume_rsi"] = (

            df["Volume"]

            *

            df["rsi"]

        )

    ####################################################
    # AGGRESSIVE BUYING
    ####################################################
    # Heavy buy-side skew combined with a real volume
    # surge — distinguishes genuine aggressive buying from
    # a quiet bar that just happens to lean buy-heavy.

    df["aggressive_buying"] = (

        (df["buy_volume_ratio"] > 0.60)

        &

        (df["volume_ratio_20"] > 1.5)

    ).astype(int)

    ####################################################
    # AGGRESSIVE SELLING
    ####################################################

    df["aggressive_selling"] = (

        (df["sell_volume_ratio"] > 0.60)

        &

        (df["volume_ratio_20"] > 1.5)

    ).astype(int)

    ####################################################
    # SMART MONEY SCORE
    ####################################################

    df["smart_money_score"] = (

        df["buy_volume_ratio"]

        *

        df["volume_ratio_20"]

    )

    return df