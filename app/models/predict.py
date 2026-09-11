from __future__ import annotations

from dataclasses import dataclass

from app.config import DEFAULT_LOOKBACK_YEARS, PREDICTION_HORIZON_DAYS
from app.data import news_sentiment
from app.data.aggregator import DataAggregator, default_lookback_start, fetch_benchmark
from app.features.build_features import SIGNAL_COLUMNS, build_latest_feature_row
from app.models import prediction_log
from app.models.train import ModelBundle, load_bundle, train_ticker


@dataclass
class Prediction:
    ticker: str
    horizon_days: int
    as_of_date: str
    last_close: float
    direction: str  # "rise" | "fall"
    direction_confidence: float  # probability of the predicted direction, 0-1
    predicted_price: float
    predicted_change_pct: float
    model_metrics: dict
    data_sources: dict
    signals: dict


def predict_ticker(ticker: str, retrain_if_missing: bool = True, refresh: bool = False) -> Prediction:
    ticker = ticker.upper().strip()
    bundle: ModelBundle | None = load_bundle(ticker)
    if bundle is None:
        if not retrain_if_missing:
            raise ValueError(f"No trained model for '{ticker}' yet. Train it first.")
        bundle = train_ticker(ticker, refresh=refresh)

    start = default_lookback_start(max(1, DEFAULT_LOOKBACK_YEARS // 2))
    aggregator = DataAggregator()
    ohlcv, source_status = aggregator.fetch(ticker, start=start, use_cache=not refresh)
    benchmark_ohlcv = fetch_benchmark(start=start, use_cache=not refresh)
    sentiment = news_sentiment.fetch_daily_sentiment(ticker, start=start, use_cache=not refresh)
    latest_features = build_latest_feature_row(ohlcv, benchmark_ohlcv=benchmark_ohlcv, sentiment=sentiment)[
        bundle.feature_columns
    ]

    proba_up = float(bundle.classifier.predict_proba(latest_features)[0, 1])
    direction = "rise" if proba_up >= 0.5 else "fall"
    confidence = proba_up if direction == "rise" else 1 - proba_up

    predicted_return = float(bundle.regressor.predict(latest_features)[0])  # regressor predicts return, not price
    last_close = float(ohlcv["close"].iloc[-1])
    predicted_price = last_close * (1 + predicted_return)
    predicted_change_pct = predicted_return * 100
    as_of_date = str(ohlcv.index[-1].date())

    signals = {col: float(latest_features.iloc[0][col]) for col in SIGNAL_COLUMNS if col in latest_features.columns}

    horizon = bundle.horizon_days or PREDICTION_HORIZON_DAYS
    prediction_log.record_prediction(
        ticker=ticker,
        as_of_date=as_of_date,
        horizon_days=horizon,
        predicted_direction=direction,
        confidence=round(confidence, 4),
        predicted_price=round(predicted_price, 4),
        last_close=round(last_close, 4),
    )

    return Prediction(
        ticker=ticker,
        horizon_days=horizon,
        as_of_date=as_of_date,
        last_close=round(last_close, 4),
        direction=direction,
        direction_confidence=round(confidence, 4),
        predicted_price=round(predicted_price, 4),
        predicted_change_pct=round(predicted_change_pct, 4),
        model_metrics=bundle.metrics,
        data_sources=source_status,
        signals={k: round(v, 4) for k, v in signals.items()},
    )


@dataclass
class WatchlistEntry:
    ticker: str
    prediction: Prediction | None
    error: str | None


def predict_watchlist(tickers: list[str]) -> list[WatchlistEntry]:
    """Predicts each ticker independently — one bad/rate-limited ticker doesn't
    take down the rest of the batch."""
    entries = []
    for ticker in tickers:
        ticker = ticker.upper().strip()
        try:
            entries.append(WatchlistEntry(ticker=ticker, prediction=predict_ticker(ticker), error=None))
        except Exception as exc:  # noqa: BLE001 - isolate per-ticker failures
            entries.append(WatchlistEntry(ticker=ticker, prediction=None, error=str(exc)))
    entries.sort(key=lambda e: e.prediction.direction_confidence if e.prediction else -1, reverse=True)
    return entries
