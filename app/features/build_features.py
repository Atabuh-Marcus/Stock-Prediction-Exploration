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
    "spy_relative_return_1d",
    "spy_relative_return_5d",
    "spy_relative_return_10d",
    "market_volatility_10d",
    "news_sentiment",
    "news_sentiment_5d_avg",
    "news_volume_10d",
]

# The subset of FEATURE_COLUMNS worth showing a human directly — the rest are
# technical-indicator plumbing the model uses but that don't read well on their own.
SIGNAL_COLUMNS = [
    "rsi_14d",
    "spy_relative_return_1d",
    "spy_relative_return_5d",
    "spy_relative_return_10d",
    "market_volatility_10d",
    "news_sentiment",
    "news_sentiment_5d_avg",
    "news_volume_10d",
]


def _price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
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
    # Spot forex pairs (e.g. EURUSD=X) report zero volume from Yahoo (OTC market, no
    # exchange tape) — a constant-zero series makes pct_change/zscore divide-by-zero
    # into NaN for every row, which would wipe out the entire feature frame via the
    # dropna() in build_training_dataset/build_latest_feature_row. Neutral (0.0) in
    # that case instead of a hard failure; other columns' own warmup NaNs (e.g. the
    # 50-day SMA gap) still trim the start of the series normally.
    features["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volume_zscore = (volume - volume_mean_20) / volume_std_20.replace(0, np.nan)
    features["volume_zscore_20d"] = volume_zscore.fillna(0.0)
    return features


def _market_context_features(index: pd.DatetimeIndex, benchmark_ohlcv: pd.DataFrame | None) -> pd.DataFrame:
    """Relative strength vs. the market (SPY) and market-wide volatility. A stock's
    move relative to the broad market is generally far more informative than its
    price action in isolation. Neutral (zero) when no benchmark is available, so
    the rest of the pipeline degrades gracefully rather than breaking.
    """
    out = pd.DataFrame(index=index)
    if benchmark_ohlcv is None or benchmark_ohlcv.empty:
        out["spy_relative_return_1d"] = 0.0
        out["spy_relative_return_5d"] = 0.0
        out["spy_relative_return_10d"] = 0.0
        out["market_volatility_10d"] = 0.0
        return out

    spy_close = benchmark_ohlcv["close"].reindex(index).ffill()
    spy_return_1d = spy_close.pct_change(1)
    out["spy_relative_return_1d"] = spy_return_1d  # filled in relative to stock below
    out["spy_relative_return_5d"] = spy_close.pct_change(5)
    out["spy_relative_return_10d"] = spy_close.pct_change(10)
    out["market_volatility_10d"] = spy_return_1d.rolling(10).std()
    return out


def _sentiment_features(index: pd.DatetimeIndex, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    """Daily news sentiment (from Alpha Vantage's NEWS_SENTIMENT endpoint), smoothed
    and forward-filled — sentiment persists between news events rather than
    resetting to neutral on quiet days. Neutral (zero) when unavailable (no Alpha
    Vantage key configured, or the endpoint returned nothing).
    """
    out = pd.DataFrame(index=index)
    if sentiment is None or sentiment.empty:
        out["news_sentiment"] = 0.0
        out["news_sentiment_5d_avg"] = 0.0
        out["news_volume_10d"] = 0.0
        return out

    daily_score = sentiment["sentiment_score"].reindex(index).ffill().fillna(0.0)
    daily_count = sentiment["article_count"].reindex(index).fillna(0.0)
    out["news_sentiment"] = daily_score
    out["news_sentiment_5d_avg"] = daily_score.rolling(5).mean()
    out["news_volume_10d"] = daily_count.rolling(10).sum()
    return out


def build_feature_frame(
    ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turns raw OHLCV (+ optional SPY benchmark, + optional news sentiment) into a
    model-ready feature matrix (unlabeled)."""
    features = _price_features(ohlcv)
    market = _market_context_features(ohlcv.index, benchmark_ohlcv)
    news = _sentiment_features(ohlcv.index, sentiment)

    combined = pd.concat([features, market, news], axis=1)
    # spy_relative_return_* start as the market's own return; subtract to get excess return.
    combined["spy_relative_return_1d"] = combined["return_1d"] - combined["spy_relative_return_1d"]
    combined["spy_relative_return_5d"] = combined["return_5d"] - combined["spy_relative_return_5d"]
    combined["spy_relative_return_10d"] = combined["return_10d"] - combined["spy_relative_return_10d"]

    return combined[FEATURE_COLUMNS]


def build_training_dataset(
    ohlcv: pd.DataFrame,
    horizon_days: int = 1,
    benchmark_ohlcv: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
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
    features = build_feature_frame(ohlcv, benchmark_ohlcv, sentiment)
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


def build_latest_feature_row(
    ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature vector for the most recent date, for live prediction."""
    features = build_feature_frame(ohlcv, benchmark_ohlcv, sentiment).replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Not enough history to compute indicators (need ~50+ trading days).")
    return features.iloc[[-1]]
