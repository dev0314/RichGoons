"""
config.py

Centralized configuration for the ``features`` package.

This module contains ONLY parameters that already exist as configurable
arguments (or module-level defaults) in the feature modules themselves:

    price_features.py        -> add_price_features(df)
    candle_features.py       -> add_candle_features(df)
    lag_features.py          -> add_lag_features(df, columns, lags, ...)
    rolling_features.py      -> add_rolling_features(df, config)
    trend_features.py        -> add_trend_features(df, ema_periods, ...)
    momentum_features.py     -> add_momentum_features(df)
    volatility_features.py   -> add_volatility_features(df, rolling_window, ...)
    volume_features.py       -> add_volume_features(df)
    market_structure.py      -> add_market_structure_features(df, lookback, ...)
    time_features.py         -> add_time_features(df)

Several modules (``candle_features``, ``momentum_features``,
``volume_features``, ``time_features``) hardcode every threshold and window
internally and expose NO parameters at all. Nothing has been invented for
those modules here -- only an ``enabled`` flag is provided so the pipeline
can still turn them on/off. Changing their behavior requires editing the
module itself, per the "do not modify feature logic" constraint.

Two things below are pipeline-level *wiring* decisions rather than values
lifted verbatim from the source, and are called out explicitly so nothing
is silently assumed:

1. ``LagFeaturesConfig.columns`` -- ``add_lag_features`` has no default for
   ``columns`` (it is a required argument), so the pipeline must supply
   one. ``["Close", "Volume"]`` is used as a conservative, always-present
   default. Override freely.

2. ``ROLLING_FEATURE_CONFIG`` -- reproduces ``rolling_features.ROLLING_CONFIG``
   verbatim (same windows, same stats, same trend combinations), with one
   correction: the source's example dict keys its third column as ``"ATR"``,
   but the column actually produced by ``volatility_features.py`` is
   lowercase ``"atr"``. Since ``add_rolling_features`` does a case-sensitive
   ``df.columns`` lookup, the original example config would raise
   ``ValueError`` the moment it ran against a real pipeline. The key is
   corrected here to ``"atr"``; every window/stat/trend value is untouched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .rolling_features import ROLLING_CONFIG as _SOURCE_ROLLING_CONFIG

# =============================================================================
# price_features.py
# =============================================================================
# add_price_features(df) takes no parameters -- every window (1, 3, 6, 12, 24
# period returns) is hardcoded in the module. Nothing configurable exists.


@dataclass(frozen=True)
class PriceFeaturesConfig:
    enabled: bool = True


# =============================================================================
# candle_features.py
# =============================================================================
# add_candle_features(df) takes no parameters. All candlestick-pattern
# thresholds (marubozu wick ratio < 0.05, hammer/shooting-star 2x body,
# spinning-top 0.30/0.25/0.25) are hardcoded in the module.


@dataclass(frozen=True)
class CandleFeaturesConfig:
    enabled: bool = True


# =============================================================================
# lag_features.py
# =============================================================================
# add_lag_features(df, columns, lags=range(1, 6), create_difference=True,
#                   create_pct_change=True, create_ratio=False)
# `columns` is a required argument in the source -- see module docstring
# note (2) above for why ["Close", "Volume"] is used as the pipeline default.


@dataclass(frozen=True)
class LagFeaturesConfig:
    enabled: bool = True
    columns: Tuple[str, ...] = ("Close", "Volume")
    lags: range = range(1, 6)
    create_difference: bool = True
    create_pct_change: bool = True
    create_ratio: bool = False


# =============================================================================
# rolling_features.py
# =============================================================================
# add_rolling_features(df, config: dict) -- `config` has no default in the
# function signature, but the module defines ROLLING_CONFIG as its own
# reference/example configuration. That dict is reused verbatim below
# (see module docstring note (2) for the one key-casing correction applied).

ROLLING_FEATURE_CONFIG: Dict[str, dict] = copy.deepcopy(_SOURCE_ROLLING_CONFIG)

if "ATR" in ROLLING_FEATURE_CONFIG:
    ROLLING_FEATURE_CONFIG["atr"] = ROLLING_FEATURE_CONFIG.pop("ATR")


@dataclass(frozen=True)
class RollingFeaturesConfig:
    enabled: bool = True
    config: Dict[str, dict] = field(default_factory=lambda: ROLLING_FEATURE_CONFIG)


# =============================================================================
# trend_features.py
# =============================================================================
# add_trend_features(df, ema_periods=(10, 20, 50, 100, 200),
#                     sma_periods=(20, 50, 200), add_adx=True, add_aroon=True)


@dataclass(frozen=True)
class TrendFeaturesConfig:
    enabled: bool = True
    ema_periods: Tuple[int, ...] = (10, 20, 50, 100, 200)
    sma_periods: Tuple[int, ...] = (20, 50, 200)
    add_adx: bool = True
    add_aroon: bool = True


# =============================================================================
# momentum_features.py
# =============================================================================
# add_momentum_features(df) takes no parameters. RSI/MACD/Stochastic/ROC/
# CCI/Williams %R/Awesome Oscillator/TSI all use the `ta` library's default
# windows, and every threshold (RSI 70/30/50) is hardcoded in the module.


@dataclass(frozen=True)
class MomentumFeaturesConfig:
    enabled: bool = True


# =============================================================================
# volatility_features.py
# =============================================================================
# add_volatility_features(df, rolling_window=20,
#                          realized_vol_windows=[5, 10, 20, 50])
# Everything past these two arguments (ATR z-score window=50, BB-squeeze
# window=50/threshold=0.75, ATR-ratio window=100, compression window=100/
# quantile=0.2, etc.) is hardcoded in the module and has no exposed
# parameter, so it is intentionally left out of this config.


@dataclass(frozen=True)
class VolatilityFeaturesConfig:
    enabled: bool = True
    rolling_window: int = 20
    realized_vol_windows: Tuple[int, ...] = (5, 10, 20, 50)


# =============================================================================
# volume_features.py
# =============================================================================
# add_volume_features(df) takes no parameters. Rolling windows [5, 10, 20, 50]
# and every threshold (buy/sell imbalance 0.60, volume spike 2x, aggressive
# 1.5x) are hardcoded in the module. It optionally reads `atr` (from
# volatility_features) and `rsi` (from momentum_features) if already present,
# which is why those two modules run first in the pipeline.


@dataclass(frozen=True)
class VolumeFeaturesConfig:
    enabled: bool = True


# =============================================================================
# market_structure.py
# =============================================================================
# add_market_structure_features(df, lookback=20, swing_order=3,
#                                fvg_min_gap=0.0, liquidity_tolerance=0.001,
#                                volume_spike_mult=1.5, trend_window=10)
#
# NOTE (existing behavior, not something this config changes):
# This module requires "ema_20" / "ema_50" to already exist on `df` (used by
# the "pullback" feature), so trend_features MUST run before this module.
# It also produces "dist_high"/"dist_low"/"volume_spike" column names that
# collide with same-named columns from price_features.py / volume_features.py
# respectively -- whichever module runs last wins. This is existing behavior
# in the source and is not altered here; see feature_pipeline.py for the
# documented execution order this relies on.


@dataclass(frozen=True)
class MarketStructureFeaturesConfig:
    enabled: bool = True
    lookback: int = 20
    swing_order: int = 3
    fvg_min_gap: float = 0.0
    liquidity_tolerance: float = 0.001
    volume_spike_mult: float = 1.5
    trend_window: int = 10


# =============================================================================
# time_features.py
# =============================================================================
# add_time_features(df) takes no parameters. Session boundaries (Asia/
# London/New York hours) and cyclical-encoding periods are hardcoded.


@dataclass(frozen=True)
class TimeFeaturesConfig:
    enabled: bool = True


# =============================================================================
# Master configuration
# =============================================================================


@dataclass(frozen=True)
class FeatureConfig:
    """
    Aggregates every per-module configuration into a single object that
    ``feature_pipeline.FeaturePipeline`` consumes.
    """

    price: PriceFeaturesConfig = field(default_factory=PriceFeaturesConfig)
    candle: CandleFeaturesConfig = field(default_factory=CandleFeaturesConfig)
    time: TimeFeaturesConfig = field(default_factory=TimeFeaturesConfig)
    trend: TrendFeaturesConfig = field(default_factory=TrendFeaturesConfig)
    momentum: MomentumFeaturesConfig = field(default_factory=MomentumFeaturesConfig)
    volatility: VolatilityFeaturesConfig = field(default_factory=VolatilityFeaturesConfig)
    volume: VolumeFeaturesConfig = field(default_factory=VolumeFeaturesConfig)
    market_structure: MarketStructureFeaturesConfig = field(
        default_factory=MarketStructureFeaturesConfig
    )
    lag: LagFeaturesConfig = field(default_factory=LagFeaturesConfig)
    rolling: RollingFeaturesConfig = field(default_factory=RollingFeaturesConfig)


DEFAULT_CONFIG = FeatureConfig()