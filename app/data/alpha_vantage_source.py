from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from app.config import ALPHA_VANTAGE_API_KEY
from app.data.base import OHLCV_COLUMNS, DataSource, DataSourceUnavailable

API_URL = "https://www.alphavantage.co/query"


class AlphaVantageSource(DataSource):
    """Free tier requires a key from https://www.alphavantage.co/support/#api-key
    (25 requests/day on the free plan). Skipped automatically if unconfigured.

    The free tier only allows outputsize=compact (last ~100 trading days,
    roughly 5 months) — outputsize=full is a paid-plan feature. So this source
    contributes recent history only; it's fine as one input to the aggregator's
    median, but won't cover a multi-year training lookback on its own.
    """

    name = "alpha_vantage"

    def is_configured(self) -> bool:
        return bool(ALPHA_VANTAGE_API_KEY)

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if not self.is_configured():
            raise DataSourceUnavailable("ALPHAVANTAGE_API_KEY is not set.")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(API_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if "Note" in payload or "Information" in payload:
            raise DataSourceUnavailable(
                f"Alpha Vantage rate-limited or restricted: {payload.get('Note') or payload.get('Information')}"
            )
        series = payload.get("Time Series (Daily)")
        if not series:
            raise DataSourceUnavailable(f"Alpha Vantage returned no data for '{ticker}': {payload}")

        df = pd.DataFrame.from_dict(series, orient="index")
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df[OHLCV_COLUMNS].astype(float).sort_index()
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if df.empty:
            raise DataSourceUnavailable(f"Alpha Vantage had no data for '{ticker}' in range.")
        return df
