"""
time_features.py

Time-based Feature Engineering

Author: RichGoons Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Open Time"
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


def add_time_features(df):

    _validate(df)

    df = df.copy()

    ####################################################
    # Convert to datetime
    ####################################################

    df["Open Time"] = pd.to_datetime(
        df["Open Time"]
    )

    ####################################################
    # Calendar Features
    ####################################################

    df["year"] = df["Open Time"].dt.year

    df["quarter"] = df["Open Time"].dt.quarter

    df["month"] = df["Open Time"].dt.month

    df["week"] = df["Open Time"].dt.isocalendar().week.astype(int)

    df["day"] = df["Open Time"].dt.day

    df["day_of_week"] = df["Open Time"].dt.dayofweek

    df["day_of_year"] = df["Open Time"].dt.dayofyear

    ####################################################
    # Intraday
    ####################################################

    df["hour"] = df["Open Time"].dt.hour

    df["minute"] = df["Open Time"].dt.minute

    ####################################################
    # Weekend
    ####################################################

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    ####################################################
    # Month Boundaries
    ####################################################

    df["month_start"] = (
        df["Open Time"].dt.is_month_start
    ).astype(int)

    df["month_end"] = (
        df["Open Time"].dt.is_month_end
    ).astype(int)

    ####################################################
    # Quarter Boundaries
    ####################################################

    df["quarter_start"] = (
        df["Open Time"].dt.is_quarter_start
    ).astype(int)

    df["quarter_end"] = (
        df["Open Time"].dt.is_quarter_end
    ).astype(int)

    ####################################################
    # Trading Sessions (UTC)
    ####################################################

    hour = df["hour"]

    df["asia_session"] = (
        (hour >= 0) &
        (hour < 8)
    ).astype(int)

    df["london_session"] = (
        (hour >= 8) &
        (hour < 16)
    ).astype(int)

    df["newyork_session"] = (
        (hour >= 13) &
        (hour < 21)
    ).astype(int)

    df["london_ny_overlap"] = (
        (hour >= 13) &
        (hour < 16)
    ).astype(int)

    ####################################################
    # Session Labels
    ####################################################

    df["session"] = np.select(
        [
            df["asia_session"] == 1,
            df["london_session"] == 1,
            df["newyork_session"] == 1,
        ],
        [
            "Asia",
            "London",
            "NewYork",
        ],
        default="Other",
    )

    ####################################################
    # Time Cyclical Encoding
    ####################################################

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    ####################################################
    # Session Open Flags
    ####################################################

    df["asia_open"] = (hour == 0).astype(int)

    df["london_open"] = (hour == 8).astype(int)

    df["ny_open"] = (hour == 13).astype(int)

    ####################################################
    # Time Since Midnight
    ####################################################

    df["minutes_since_midnight"] = (
        df["hour"] * 60 +
        df["minute"]
    )

    return df