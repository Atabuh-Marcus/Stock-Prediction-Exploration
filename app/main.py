from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.data.aggregator import DataAggregator, default_lookback_start
from app.data.base import DataSourceUnavailable
from app.models import prediction_log
from app.models.backtest import run_backtest
from app.models.predict import predict_ticker, predict_watchlist
from app.models.train import train_ticker

app = FastAPI(
    title="Stock Direction Prediction API",
    description="Predicts the next trading day's direction and price from historical market data "
    "combined across multiple providers.",
    version="0.2.0",
)

TICKER_QUERY = Query(..., min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-]+$")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class PredictionResponse(BaseModel):
    symbol: str
    direction: Literal["rise", "fall"]
    direction_confidence: float = Field(ge=0, le=1)
    last_close: float
    predicted_price: float
    predicted_change_pct: float
    horizon_days: int
    as_of: str
    model_metrics: dict
    data_sources: dict
    signals: dict
    note: str = "Model output is informational, not financial advice."


class TrainResponse(BaseModel):
    symbol: str
    trained_at: str
    metrics: dict
    data_sources: dict


class HistoryPoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoryResponse(BaseModel):
    symbol: str
    data_sources: dict
    points: list[HistoryPoint]


class WatchlistEntryResponse(BaseModel):
    symbol: str
    prediction: PredictionResponse | None
    error: str | None


class BacktestResponse(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    evaluated_days: int
    retrains: int
    confidence_threshold: float
    hit_rate: float
    strategy_total_return_pct: float
    buy_hold_total_return_pct: float
    strategy_sharpe: float
    strategy_max_drawdown_pct: float
    days_in_market_pct: float
    equity_curve: list[dict]
    note: str


class PredictionLogRow(BaseModel):
    ticker: str
    as_of_date: str
    target_date: str
    horizon_days: int
    predicted_direction: str
    confidence: float
    predicted_price: float
    last_close: float
    resolved: bool
    actual_close: float | None
    actual_direction: str | None
    correct: bool | None
    realized_change_pct: float | None


class CalibrationBucket(BaseModel):
    range: str
    count: int
    actual_accuracy: float


class PredictionHistoryResponse(BaseModel):
    total_predictions: int
    resolved_predictions: int
    accuracy: float | None
    calibration_buckets: list[CalibrationBucket]
    rows: list[PredictionLogRow]


class ComparePoint(BaseModel):
    date: date
    pct_change: float


class CompareSeriesResponse(BaseModel):
    symbol: str
    points: list[ComparePoint]
    error: str | None


class CompareResponse(BaseModel):
    series: list[CompareSeriesResponse]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _to_prediction_response(result) -> PredictionResponse:
    return PredictionResponse(
        symbol=result.ticker,
        direction=result.direction,
        direction_confidence=result.direction_confidence,
        last_close=result.last_close,
        predicted_price=result.predicted_price,
        predicted_change_pct=result.predicted_change_pct,
        horizon_days=result.horizon_days,
        as_of=result.as_of_date,
        model_metrics=result.model_metrics,
        data_sources=result.data_sources,
        signals=result.signals,
    )


@app.get("/predict", response_model=PredictionResponse)
def predict(symbol: str = TICKER_QUERY, refresh: bool = Query(False)) -> PredictionResponse:
    normalized_symbol = symbol.upper()
    try:
        result = predict_ticker(normalized_symbol, refresh=refresh)
        return _to_prediction_response(result)
    except (ValueError, DataSourceUnavailable) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Market data provider error: {error}") from error


@app.get("/watchlist", response_model=list[WatchlistEntryResponse])
def watchlist(symbols: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL")) -> list[
    WatchlistEntryResponse
]:
    tickers = [t.strip().upper() for t in symbols.split(",") if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="Provide at least one ticker in `symbols`.")
    if len(tickers) > 20:
        raise HTTPException(status_code=422, detail="Max 20 tickers per watchlist request.")

    entries = predict_watchlist(tickers)
    return [
        WatchlistEntryResponse(
            symbol=e.ticker,
            prediction=_to_prediction_response(e.prediction) if e.prediction else None,
            error=e.error,
        )
        for e in entries
    ]


@app.post("/train", response_model=TrainResponse)
def train(
    symbol: str = TICKER_QUERY,
    lookback_years: int = Query(5, ge=1, le=20),
    horizon_days: int = Query(1, ge=1, le=30),
    refresh: bool = Query(False),
) -> TrainResponse:
    normalized_symbol = symbol.upper()
    try:
        bundle = train_ticker(
            normalized_symbol, lookback_years=lookback_years, horizon_days=horizon_days, refresh=refresh
        )
        return TrainResponse(
            symbol=bundle.ticker,
            trained_at=bundle.trained_at,
            metrics=bundle.metrics,
            data_sources=bundle.source_status,
        )
    except (ValueError, DataSourceUnavailable) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Market data provider error: {error}") from error


@app.get("/backtest", response_model=BacktestResponse)
def backtest(
    symbol: str = TICKER_QUERY,
    lookback_years: int = Query(5, ge=1, le=20),
    horizon_days: int = Query(1, ge=1, le=30),
    retrain_every_days: int = Query(20, ge=1, le=250),
    confidence_threshold: float = Query(0.5, ge=0.5, le=0.99),
    refresh: bool = Query(False),
) -> BacktestResponse:
    normalized_symbol = symbol.upper()
    try:
        result = run_backtest(
            normalized_symbol,
            lookback_years=lookback_years,
            horizon_days=horizon_days,
            retrain_every_days=retrain_every_days,
            confidence_threshold=confidence_threshold,
            refresh=refresh,
        )
        return BacktestResponse(
            symbol=result.ticker,
            start_date=result.start_date,
            end_date=result.end_date,
            evaluated_days=result.evaluated_days,
            retrains=result.retrains,
            confidence_threshold=result.confidence_threshold,
            hit_rate=result.hit_rate,
            strategy_total_return_pct=result.strategy_total_return_pct,
            buy_hold_total_return_pct=result.buy_hold_total_return_pct,
            strategy_sharpe=result.strategy_sharpe,
            strategy_max_drawdown_pct=result.strategy_max_drawdown_pct,
            days_in_market_pct=result.days_in_market_pct,
            equity_curve=result.equity_curve,
            note=result.note,
        )
    except (ValueError, DataSourceUnavailable) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Market data provider error: {error}") from error


@app.get("/history", response_model=HistoryResponse)
def history(symbol: str = TICKER_QUERY, days: int = Query(180, ge=5, le=3650)) -> HistoryResponse:
    normalized_symbol = symbol.upper()
    try:
        aggregator = DataAggregator()
        ohlcv, status = aggregator.fetch(normalized_symbol, start=default_lookback_start(1))
        recent = ohlcv.tail(days)
        points = [
            HistoryPoint(
                date=row.Index.date(),
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in recent.itertuples(index=True, name="Row")
        ]
        return HistoryResponse(symbol=normalized_symbol, data_sources=status, points=points)
    except (ValueError, DataSourceUnavailable) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Market data provider error: {error}") from error


@app.get("/predictions/history", response_model=PredictionHistoryResponse)
def predictions_history(
    symbol: str | None = Query(None, min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-]+$")
) -> PredictionHistoryResponse:
    result = prediction_log.get_history(symbol.upper() if symbol else None)
    return PredictionHistoryResponse(**result)


@app.get("/compare", response_model=CompareResponse)
def compare(
    symbols: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,SPY"),
    days: int = Query(180, ge=5, le=3650),
) -> CompareResponse:
    tickers = [t.strip().upper() for t in symbols.split(",") if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="Provide at least one ticker in `symbols`.")
    if len(tickers) > 10:
        raise HTTPException(status_code=422, detail="Max 10 tickers per comparison request.")

    aggregator = DataAggregator()
    series_list = []
    for ticker in tickers:
        try:
            ohlcv, _ = aggregator.fetch(ticker, start=default_lookback_start(1))
            recent = ohlcv["close"].tail(days)
            base = recent.iloc[0]
            pct_change = (recent / base - 1) * 100
            points = [ComparePoint(date=idx.date(), pct_change=round(float(v), 4)) for idx, v in pct_change.items()]
            series_list.append(CompareSeriesResponse(symbol=ticker, points=points, error=None))
        except Exception as exc:  # noqa: BLE001 - isolate per-ticker failures, matching /watchlist
            series_list.append(CompareSeriesResponse(symbol=ticker, points=[], error=str(exc)))

    return CompareResponse(series=series_list)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
