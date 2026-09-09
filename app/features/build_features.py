from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.indicators import bollinger_bandwidth, ema, macd, rsi, sma

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "sma_gap_5d",
    "sma_gap_10d",
    "sma_gap_20d",
    "sma_gap_50d",
    "ema_gap_12d",
    "rsi_14d",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_bandwidth",
    "volatility_10d",
    "volatility_20d",
    "volume_change",
    "volume_zscore_20d",
]


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Turns raw OHLCV into a model-ready feature matrix (unlabeled)."""
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    macd_line, macd_signal_line = macd(close)
    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std()

    # MACD is an absolute dollar value by construction; dividing by price makes it
    # comparable across a stock's history even as its price level drifts over time.
    macd_norm = macd_line / close
    macd_signal_norm = macd_signal_line / close

    features = pd.DataFrame(index=ohlcv.index)
    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["sma_gap_5d"] = close / sma(close, 5) - 1
    features["sma_gap_10d"] = close / sma(close, 10) - 1
    features["sma_gap_20d"] = close / sma(close, 20) - 1
    features["sma_gap_50d"] = close / sma(close, 50) - 1
    features["ema_gap_12d"] = close / ema(close, 12) - 1
    features["rsi_14d"] = rsi(close, 14)
    features["macd"] = macd_norm
    features["macd_signal"] = macd_signal_norm
    features["macd_hist"] = macd_norm - macd_signal_norm
    features["bollinger_bandwidth"] = bollinger_bandwidth(close, 20)
    features["volatility_10d"] = close.pct_change().rolling(10).std()
    features["volatility_20d"] = close.pct_change().rolling(20).std()
    features["volume_change"] = volume.pct_change()
    features["volume_zscore_20d"] = (volume - volume_mean_20) / volume_std_20.replace(0, np.nan)

    return features[FEATURE_COLUMNS]


def build_training_dataset(
    ohlcv: pd.DataFrame, horizon_days: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Builds (features, classification_target, regression_target) aligned and
    trimmed so every row has both a full feature vector and a known future label.

    classification_target: 1 if close rises `horizon_days` ahead, else 0.
    regression_target: the % return `horizon_days` ahead (not the absolute price —
    the feature set is almost entirely scale-invariant, so a model trained to
    predict absolute price has no way to anchor its output near the current price;
    predicting the return and reconstructing price = last_close * (1 + return)
    keeps predictions calibrated to where the stock actually is).
    """
    features = build_feature_frame(ohlcv)
    close = ohlcv["close"]

    future_close = close.shift(-horizon_days)
    classification_target = (future_close > close).astype(int)
    regression_target = (future_close / close) - 1

    valid = features.replace([np.inf, -np.inf], np.nan).dropna().index
    valid = valid.intersection(regression_target.dropna().index)

    return (
        features.loc[valid],
        classification_target.loc[valid],
        regression_target.loc[valid],
    )


def build_latest_feature_row(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature vector for the most recent date, for live prediction."""
    features = build_feature_frame(ohlcv).replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Not enough history to compute indicators (need ~50+ trading days).")
    return features.iloc[[-1]]
