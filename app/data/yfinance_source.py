from __future__ import annotations

import time
from datetime import date

import pandas as pd
import yfinance as yf

from app.data.base import OHLCV_COLUMNS, DataSource, DataSourceUnavailable

RENAME_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceSource(DataSource):
    """Free, no API key required. Good default / always-on source.

    Yahoo's endpoints are flaky about silently returning an empty frame (bot
    detection, transient rate limiting) rather than raising, so this tries two
    different yfinance call paths and retries each once before giving up.
    """

    name = "yfinance"

    def is_configured(self) -> bool:
        return True

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        attempts = [
            lambda: yf.Ticker(ticker).history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=True
            ),
            lambda: yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            ),
        ]

        last_error: str | None = None
        for fetch in attempts:
            for retry in range(2):
                try:
                    raw = fetch()
                except Exception as exc:  # noqa: BLE001 - fall through to next attempt/retry
                    last_error = str(exc)
                    raw = None
                else:
                    if raw is not None and not raw.empty:
                        return self._normalize(raw)
                    last_error = "provider returned an empty result (likely bot-detection/rate-limiting)"
                if retry == 0:
                    time.sleep(1.5)

        detail = f" Last error: {last_error}" if last_error else ""
        raise DataSourceUnavailable(
            f"yfinance returned no data for '{ticker}' after retrying two call paths. "
            f"This is usually transient Yahoo rate-limiting — try again in a minute, or "
            f"run `pip install --upgrade yfinance`.{detail}"
        )

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=RENAME_MAP)

        missing = set(OHLCV_COLUMNS) - set(raw.columns)
        if missing:
            raise DataSourceUnavailable(f"yfinance response missing columns: {missing}")

        df = raw[OHLCV_COLUMNS].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        return df
