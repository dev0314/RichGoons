"""
feature_pipeline.py

Orchestrates every feature module in ``features/`` behind a single
``fit`` / ``transform`` / ``fit_transform`` interface.

Execution order
----------------
The order below is derived from the actual dependencies found in the
source modules, not assumed:

1. price_features        -- needs only Open/High/Low/Close.
2. candle_features        -- needs only Open/High/Low/Close.
3. time_features          -- needs only "Open Time".
4. trend_features          -- needs High/Low/Close. Produces "ema_20" and
                              "ema_50", which market_structure requires.
5. momentum_features       -- needs High/Low/Close. Produces "rsi", which
                              volume_features reads if present.
6. volatility_features     -- needs High/Low/Close. Produces "atr", which
                              both volume_features and the rolling-features
                              config read if present.
7. volume_features         -- needs OHLCV + taker-volume columns. Reads
                              "atr" (step 6) and "rsi" (step 5) if present
                              to build the volume_atr / volume_rsi
                              interaction features -- hence it runs after
                              both.
8. market_structure        -- REQUIRES "ema_20" / "ema_50" (step 4) for its
                              "pullback" feature; must run after
                              trend_features. NOTE: this module's own
                              "dist_high"/"dist_low"/"volume_spike" columns
                              overwrite the same-named columns produced by
                              price_features (step 1) and volume_features
                              (step 7) respectively. This is existing
                              behavior in the source, not something this
                              pipeline changes -- see config.py for details.
9. lag_features            -- generic; lags whatever columns are configured
                              (default: Close, Volume), so it can safely run
                              after every column-producing module.
10. rolling_features        -- generic; the default config keys a column
                              named "atr", so it must run after step 6.

Because market_structure depends on trend_features output, and
volume_features / rolling_features optionally depend on volatility_features
/ momentum_features output, this order cannot be shuffled without either
raising a KeyError or silently losing the cross-module interaction features.

Design notes
------------
Every feature computed here is a deterministic function of the input
OHLCV(+time) data -- there are no learned parameters (no scalers, no
fitted statistics) anywhere in the existing modules. Accordingly:

* ``fit`` validates that the configured base columns exist and returns
  ``self``. It does not learn anything, because there is nothing in the
  source modules to learn.
* ``transform`` runs the configured modules in the order above and
  returns the enriched DataFrame.
* ``fit_transform`` is ``fit`` followed by ``transform``, matching the
  common scikit-learn-style convention.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import DEFAULT_CONFIG, FeatureConfig
from .price_features import add_price_features
from .candle_features import add_candle_features
from .time_features import add_time_features
from .trend_features import add_trend_features
from .momentum_features import add_momentum_features
from .volatility_features import add_volatility_features
from .volume_features import add_volume_features
from .market_structure import add_market_structure_features
from .lag_features import add_lag_features
from .rolling_features import add_rolling_features

logger = logging.getLogger(__name__)


# Columns required somewhere in the pipeline, keyed by the module that
# requires them. Used by `fit` to fail fast with a clear error instead of
# partway through `transform`.
_BASE_COLUMN_REQUIREMENTS = {
    "price": ["Open", "High", "Low", "Close"],
    "candle": ["Open", "High", "Low", "Close"],
    "time": ["Open Time"],
    "trend": ["High", "Low", "Close"],
    "momentum": ["High", "Low", "Close"],
    "volatility": ["High", "Low", "Close"],
    "volume": [
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Volume",
        "Taker Buy Quote Volume",
    ],
    "market_structure": ["High", "Low", "Close", "Volume"],
}

# Fixed dependency-respecting execution order (see module docstring).
_EXECUTION_ORDER = (
    "price",
    "candle",
    "time",
    "trend",
    "momentum",
    "volatility",
    "volume",
    "market_structure",
    "lag",
    "rolling",
)


class FeaturePipeline:
    """
    Runs every ``add_*_features`` function in ``features/`` in the correct
    dependency order, with each module individually enabled/disabled and
    configured through a :class:`~features.config.FeatureConfig`.

    Parameters
    ----------
    config : FeatureConfig, optional
        Defaults to ``features.config.DEFAULT_CONFIG``.

    Examples
    --------
    >>> pipeline = FeaturePipeline()
    >>> enriched = pipeline.fit_transform(df)

    >>> from features.config import FeatureConfig, TrendFeaturesConfig
    >>> custom = FeatureConfig(trend=TrendFeaturesConfig(ema_periods=(20, 50)))
    >>> pipeline = FeaturePipeline(config=custom)
    >>> enriched = pipeline.fit(df).transform(df)
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config: FeatureConfig = config or DEFAULT_CONFIG
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # fit / transform / fit_transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Validate that the base columns required by each *enabled* module
        are present in ``df``. No statistics are learned -- every feature
        module in this package is a stateless, deterministic transform of
        its input.

        Raises
        ------
        ValueError
            If a required column for an enabled module is missing.
        """

        missing_report = {}

        for step in _EXECUTION_ORDER:

            if not self._module_enabled(step):
                continue

            required = _BASE_COLUMN_REQUIREMENTS.get(step)

            if not required:
                continue

            missing = [c for c in required if c not in df.columns]

            if missing:
                missing_report[step] = missing

        if missing_report:
            raise ValueError(
                f"Missing required base columns for enabled modules: {missing_report}"
            )

        # lag_features additionally requires its configured columns to
        # exist. They are typically produced by earlier modules, so this
        # is checked against the *input* df only as an early sanity check
        # for columns that are expected to already exist pre-pipeline
        # (e.g. "Close", "Volume"); columns produced mid-pipeline (e.g.
        # "rsi", "atr") are validated at transform time instead.
        if self._module_enabled("lag"):
            pre_existing_lag_cols = [
                c for c in self.config.lag.columns if c in df.columns
            ]
            configured_but_absent = [
                c
                for c in self.config.lag.columns
                if c not in df.columns and c not in _KNOWN_DERIVED_COLUMNS
            ]
            if configured_but_absent:
                logger.warning(
                    "lag_features configured to lag columns not present in the "
                    "input and not recognized as a derived feature column: %s. "
                    "transform() will raise if they are not produced by an "
                    "earlier step.",
                    configured_but_absent,
                )

        self._is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run every enabled feature module, in dependency order, and return
        the enriched DataFrame. Does not mutate ``df`` in place.
        """

        if not self._is_fitted:
            raise RuntimeError("FeaturePipeline.transform() called before fit().")

        result = df.copy()

        for step in _EXECUTION_ORDER:

            if not self._module_enabled(step):
                logger.debug("Skipping disabled module: %s", step)
                continue

            logger.debug("Running module: %s", step)

            result = self._run_step(step, result)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equivalent to ``self.fit(df).transform(df)``."""

        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _module_enabled(self, step: str) -> bool:
        return bool(getattr(self.config, step).enabled)

    def _run_step(self, step: str, df: pd.DataFrame) -> pd.DataFrame:

        if step == "price":
            return add_price_features(df)

        if step == "candle":
            return add_candle_features(df)

        if step == "time":
            return add_time_features(df)

        if step == "trend":
            cfg = self.config.trend
            return add_trend_features(
                df,
                ema_periods=cfg.ema_periods,
                sma_periods=cfg.sma_periods,
                add_adx=cfg.add_adx,
                add_aroon=cfg.add_aroon,
            )

        if step == "momentum":
            return add_momentum_features(df)

        if step == "volatility":
            cfg = self.config.volatility
            return add_volatility_features(
                df,
                rolling_window=cfg.rolling_window,
                realized_vol_windows=list(cfg.realized_vol_windows),
            )

        if step == "volume":
            return add_volume_features(df)

        if step == "market_structure":
            cfg = self.config.market_structure
            return add_market_structure_features(
                df,
                lookback=cfg.lookback,
                swing_order=cfg.swing_order,
                fvg_min_gap=cfg.fvg_min_gap,
                liquidity_tolerance=cfg.liquidity_tolerance,
                volume_spike_mult=cfg.volume_spike_mult,
                trend_window=cfg.trend_window,
            )

        if step == "lag":
            cfg = self.config.lag
            return add_lag_features(
                df,
                columns=list(cfg.columns),
                lags=cfg.lags,
                create_difference=cfg.create_difference,
                create_pct_change=cfg.create_pct_change,
                create_ratio=cfg.create_ratio,
            )

        if step == "rolling":
            cfg = self.config.rolling
            return add_rolling_features(df, config=cfg.config)

        raise ValueError(f"Unknown pipeline step: {step}")


# Columns produced mid-pipeline that are valid lag targets even though they
# are absent from the raw input DataFrame. Used only to keep fit()'s
# early sanity-check warning free of false positives for the default config.
_KNOWN_DERIVED_COLUMNS = {
    "rsi",
    "atr",
    "atr_pct",
    "macd",
    "macd_hist",
    "obv",
    "cmf",
    "mfi",
    "force_index",
    "adx",
    "true_range",
    "historical_volatility",
}