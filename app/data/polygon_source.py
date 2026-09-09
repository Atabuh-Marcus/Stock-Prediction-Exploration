from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from app.config import POLYGON_API_KEY
from app.data.base import OHLCV_COLUMNS, DataSource, DataSourceUnavailable

API_URL_TEMPLATE = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"


class PolygonSource(DataSource):
    """Paid/production-grade source. Requires POLYGON_API_KEY. Skipped automatically
    if unconfigured, so the app still works without it.
    """

    name = "polygon"

    def is_configured(self) -> bool:
        return bool(POLYGON_API_KEY)

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if not self.is_configured():
            raise DataSourceUnavailable("POLYGON_API_KEY is not set.")

        url = API_URL_TEMPLATE.format(ticker=ticker, start=start.isoformat(), end=end.isoformat())
        response = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results")
        if not results:
            raise DataSourceUnavailable(f"Polygon returned no data for '{ticker}': {payload.get('status')}")

        df = pd.DataFrame(results)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "ts"})
        df.index = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df.index.name = "date"
        return df[OHLCV_COLUMNS].astype(float).sort_index()
