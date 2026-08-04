"""
trend_features.py

Trend based feature engineering.

Author: RichGoons Project
"""

from __future__ import annotations

import pandas as pd

from ta.trend import EMAIndicator
from ta.trend import SMAIndicator
from ta.trend import ADXIndicator
from ta.trend import AroonIndicator


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
            f"Missing columns {missing}"
        )


def add_trend_features(
    df: pd.DataFrame,
    ema_periods=(10,20,50,100,200),
    sma_periods=(20,50,200),
    add_adx=True,
    add_aroon=True,
):

    _validate(df)

    df = df.copy()

    #######################################################
    # EMA
    #######################################################

    for period in ema_periods:

        ema = EMAIndicator(
            close=df["Close"],
            window=period
        )

        df[f"ema_{period}"] = (
            ema.ema_indicator()
        )

    #######################################################
    # SMA
    #######################################################

    for period in sma_periods:

        sma = SMAIndicator(
            close=df["Close"],
            window=period
        )

        df[f"sma_{period}"] = (
            sma.sma_indicator()
        )

    #######################################################
    # DISTANCE FROM EMA
    #######################################################

    for period in ema_periods:

        ema = df[f"ema_{period}"]

        df[f"ema_dist_{period}"] = (
            df["Close"] - ema
        )

        df[f"ema_dist_pct_{period}"] = (
            (df["Close"] - ema)
            / ema
        )

        df[f"ema_slope_{period}"] = (
            ema.diff()
        )

    #######################################################
    # DISTANCE FROM SMA
    #######################################################

    for period in sma_periods:

        sma = df[f"sma_{period}"]

        df[f"sma_dist_{period}"] = (
            df["Close"] - sma
        )

        df[f"sma_slope_{period}"] = (
            sma.diff()
        )

    #######################################################
    # EMA SPREADS
    #######################################################

    spreads = [

        (10,20),

        (20,50),

        (50,100),

        (50,200),

        (100,200)

    ]

    for fast, slow in spreads:

        if (
            fast in ema_periods
            and
            slow in ema_periods
        ):

            df[f"ema_spread_{fast}_{slow}"] = (

                df[f"ema_{fast}"]

                -

                df[f"ema_{slow}"]

            )

            df[f"ema_spread_pct_{fast}_{slow}"] = (

                (

                    df[f"ema_{fast}"]

                    -

                    df[f"ema_{slow}"]

                )

                /

                df[f"ema_{slow}"]

            )

    #######################################################
    # PRICE ABOVE EMA
    #######################################################

    for period in ema_periods:

        df[f"price_above_ema_{period}"] = (

            df["Close"]

            >

            df[f"ema_{period}"]

        ).astype(int)

    #######################################################
    # EMA ALIGNMENT
    #######################################################

    if all(
        p in ema_periods
        for p in [10,20,50,200]
    ):

        df["bullish_alignment"] = (

            (df["ema_10"] > df["ema_20"])

            &

            (df["ema_20"] > df["ema_50"])

            &

            (df["ema_50"] > df["ema_200"])

        ).astype(int)

        df["bearish_alignment"] = (

            (df["ema_10"] < df["ema_20"])

            &

            (df["ema_20"] < df["ema_50"])

            &

            (df["ema_50"] < df["ema_200"])

        ).astype(int)

    #######################################################
    # ADX
    #######################################################

    if add_adx:

        adx = ADXIndicator(

            high=df["High"],

            low=df["Low"],

            close=df["Close"]

        )

        df["adx"] = adx.adx()

        df["plus_di"] = adx.adx_pos()

        df["minus_di"] = adx.adx_neg()

        df["strong_trend"] = (

            df["adx"] > 25

        ).astype(int)

    #######################################################
    # AROON
    #######################################################

    if add_aroon:

        aroon = AroonIndicator(

            high=df["High"],

            low=df["Low"]

        )

        df["aroon_up"] = (

            aroon.aroon_up()

        )

        df["aroon_down"] = (

            aroon.aroon_down()

        )

    return df