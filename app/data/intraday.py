from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.data.base import DataSourceUnavailable

# yfinance only — the "live" chart doesn't need the daily aggregator's multi-source
# median or its day-level cache; it needs the freshest bars it can get, and none of
# Alpha Vantage's/Polygon's free tiers offer meaningful free intraday history anyway.
VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m"}

# yfinance only keeps a few days of the shortest intervals — request just enough
# history for a readable intraday chart without asking for a range it will refuse.
_PERIOD_BY_INTERVAL = {
    "1m": "1d",
    "2m": "5d",
    "5m": "5d",
    "15m": "5d",
    "30m": "1mo",
    "60m": "1mo",
}


def fetch_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}")

    period = _PERIOD_BY_INTERVAL[interval]
    try:
        raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean provider error below
        raise DataSourceUnavailable(f"yfinance intraday fetch failed for '{ticker}': {exc}") from exc

    if raw is None or raw.empty:
        raise DataSourceUnavailable(
            f"No intraday data for '{ticker}' at {interval} — markets may be closed, or this "
            f"symbol/interval combination isn't available from Yahoo Finance."
        )

    raw = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "time"
    return df
