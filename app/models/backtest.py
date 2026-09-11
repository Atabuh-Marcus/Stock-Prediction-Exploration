from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import DEFAULT_LOOKBACK_YEARS, PREDICTION_HORIZON_DAYS
from app.data import news_sentiment
from app.data.aggregator import DataAggregator, default_lookback_start, fetch_benchmark
from app.features.build_features import build_training_dataset
from app.models import model_factory

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    evaluated_days: int
    retrains: int
    confidence_threshold: float
    hit_rate: float  # walk-forward directional accuracy, out-of-sample throughout
    strategy_total_return_pct: float
    buy_hold_total_return_pct: float
    strategy_sharpe: float
    strategy_max_drawdown_pct: float
    days_in_market_pct: float
    equity_curve: list[dict] = field(default_factory=list)
    note: str = (
        "Simulates a simple long/cash strategy: long when predicted direction is 'rise' "
        "with confidence >= threshold, otherwise flat. No shorting, fees, or slippage "
        "modeled. Educational only, not investment advice."
    )


def run_backtest(
    ticker: str,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
    retrain_every_days: int = 20,
    min_train_rows: int = 250,
    confidence_threshold: float = 0.5,
    refresh: bool = False,
) -> BacktestResult:
    """Walk-forward backtest: retrains periodically using only data available up
    to that point (no lookahead), then simulates trading on the following block
    with that fixed model before retraining again.
    """
    ticker = ticker.upper().strip()
    start = default_lookback_start(lookback_years)
    aggregator = DataAggregator()
    ohlcv, _ = aggregator.fetch(ticker, start=start, use_cache=not refresh)
    benchmark_ohlcv = fetch_benchmark(start=start, use_cache=not refresh)
    sentiment = news_sentiment.fetch_daily_sentiment(ticker, start=start, use_cache=not refresh)

    features, class_target, reg_target = build_training_dataset(
        ohlcv, horizon_days, benchmark_ohlcv=benchmark_ohlcv, sentiment=sentiment
    )
    n = len(features)
    if n < min_train_rows + retrain_every_days:
        raise ValueError(
            f"Only {n} usable rows for '{ticker}' — need at least "
            f"{min_train_rows + retrain_every_days} for a walk-forward backtest. Try a longer lookback."
        )

    predicted_direction = np.empty(n - min_train_rows, dtype=bool)
    predicted_confidence = np.empty(n - min_train_rows, dtype=float)
    retrains = 0

    eval_start = min_train_rows
    checkpoint = eval_start

    # Hyperparameters are tuned once, on the initial window only (no lookahead —
    # this is the same data the very first retrain checkpoint trains on anyway),
    # then reused at every subsequent retrain. Re-running a full search at each of
    # the (potentially dozens of) walk-forward checkpoints would be far too slow;
    # this still gets tuned settings instead of hard-coded defaults, just not
    # re-tuned as the window grows.
    classifier_params = model_factory.tune_classifier(features.iloc[:eval_start], class_target.iloc[:eval_start])

    while checkpoint < n:
        block_end = min(checkpoint + retrain_every_days, n)
        train_X = features.iloc[:checkpoint]
        train_y_class = class_target.iloc[:checkpoint]

        classifier = model_factory.fit_calibrated_classifier(train_X, train_y_class, classifier_params)
        retrains += 1

        block_X = features.iloc[checkpoint:block_end]
        proba_up = classifier.predict_proba(block_X)[:, 1]
        offset = checkpoint - eval_start
        predicted_direction[offset : offset + len(block_X)] = proba_up >= 0.5
        predicted_confidence[offset : offset + len(block_X)] = np.where(proba_up >= 0.5, proba_up, 1 - proba_up)

        checkpoint = block_end

    eval_index = features.index[eval_start:]
    actual_direction = class_target.iloc[eval_start:].to_numpy().astype(bool)
    actual_return = reg_target.iloc[eval_start:].to_numpy()  # realized forward return, horizon_days ahead

    in_market = predicted_direction & (predicted_confidence >= confidence_threshold)
    strategy_return = np.where(in_market, actual_return, 0.0)

    strategy_cum = np.cumprod(1 + strategy_return) - 1
    buy_hold_cum = np.cumprod(1 + actual_return) - 1

    hit_rate = float(np.mean(predicted_direction == actual_direction))
    days_in_market_pct = float(np.mean(in_market)) * 100

    equity = 1 + strategy_cum
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_drawdown_pct = float(drawdown.min()) * 100

    periods_per_year = TRADING_DAYS_PER_YEAR / max(horizon_days, 1)
    strategy_std = strategy_return.std()
    sharpe = (
        float(strategy_return.mean() / strategy_std * np.sqrt(periods_per_year)) if strategy_std > 0 else 0.0
    )

    equity_curve = [
        {
            "date": str(d.date()),
            "strategy_cum_return_pct": round(float(s) * 100, 4),
            "buy_hold_cum_return_pct": round(float(b) * 100, 4),
        }
        for d, s, b in zip(eval_index, strategy_cum, buy_hold_cum)
    ]

    return BacktestResult(
        ticker=ticker,
        start_date=str(eval_index[0].date()),
        end_date=str(eval_index[-1].date()),
        evaluated_days=len(eval_index),
        retrains=retrains,
        confidence_threshold=confidence_threshold,
        hit_rate=round(hit_rate, 4),
        strategy_total_return_pct=round(float(strategy_cum[-1]) * 100, 4),
        buy_hold_total_return_pct=round(float(buy_hold_cum[-1]) * 100, 4),
        strategy_sharpe=round(sharpe, 4),
        strategy_max_drawdown_pct=round(max_drawdown_pct, 4),
        days_in_market_pct=round(days_in_market_pct, 2),
        equity_curve=equity_curve,
    )
