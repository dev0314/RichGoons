"""
features

Feature engineering package for the RichGoons project.

Exposes every public ``add_*_features`` function from the individual
feature modules, the shared configuration objects, and the
``FeaturePipeline`` orchestrator.
"""

from __future__ import annotations

from .price_features import add_price_features
from .candle_features import add_candle_features
from .lag_features import add_lag_features
from .rolling_features import add_rolling_features, ROLLING_CONFIG
from .trend_features import add_trend_features
from .momentum_features import add_momentum_features
from .volatility_features import add_volatility_features
from .xau_volatility_features import add_xau_volatility_features
from .volume_features import add_volume_features
from .market_structure import add_market_structure_features
from .time_features import add_time_features

from .config import (
    FeatureConfig,
    DEFAULT_CONFIG,
    PriceFeaturesConfig,
    CandleFeaturesConfig,
    LagFeaturesConfig,
    RollingFeaturesConfig,
    TrendFeaturesConfig,
    MomentumFeaturesConfig,
    VolatilityFeaturesConfig,
    VolumeFeaturesConfig,
    MarketStructureFeaturesConfig,
    TimeFeaturesConfig,
    ROLLING_FEATURE_CONFIG,
)

from .feature_pipeline import FeaturePipeline

__all__ = [
    # Feature functions
    "add_price_features",
    "add_candle_features",
    "add_lag_features",
    "add_rolling_features",
    "add_trend_features",
    "add_momentum_features",
    "add_volatility_features",
    "add_xau_volatility_features",
    "add_volume_features",
    "add_market_structure_features",
    "add_time_features",
    # Reference configuration data
    "ROLLING_CONFIG",
    "ROLLING_FEATURE_CONFIG",
    # Configuration objects
    "FeatureConfig",
    "DEFAULT_CONFIG",
    "PriceFeaturesConfig",
    "CandleFeaturesConfig",
    "LagFeaturesConfig",
    "RollingFeaturesConfig",
    "TrendFeaturesConfig",
    "MomentumFeaturesConfig",
    "VolatilityFeaturesConfig",
    "VolumeFeaturesConfig",
    "MarketStructureFeaturesConfig",
    "TimeFeaturesConfig",
    # Pipeline
    "FeaturePipeline",
    "XAU_VOLATILITY_FEATURES",
    
]