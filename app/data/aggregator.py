from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from app.data import cache
from app.data.alpha_vantage_source import AlphaVantageSource
from app.data.base import OHLCV_COLUMNS, DataSource, DataSourceUnavailable
from app.data.polygon_source import PolygonSource
from app.data.yfinance_source import YFinanceSource

logger = logging.getLogger(__name__)

# yfinance first: it's always configured and rarely fails, so it anchors the
# combined series even when the paid/free-tier-limited sources are unavailable.
DEFAULT_SOURCES: list[DataSource] = [YFinanceSource(), AlphaVantageSource(), PolygonSource()]


class DataAggregator:
    """Fetches OHLCV from every configured source and combines them into one series.

    Combination strategy: outer-join all sources on date, then take the per-day
    median across whichever sources returned a value for that day/column. Median
    (rather than mean) means one source glitching on a given day doesn't skew the
    combined price as long as at least one other source agrees.
    """

    def __init__(self, sources: list[DataSource] | None = None):
        self.sources = sources if sources is not None else DEFAULT_SOURCES

    def available_sources(self) -> list[str]:
        return [s.name for s in self.sources if s.is_configured()]

    def fetch(
        self, ticker: str, start: date, end: date | None = None, use_cache: bool = True
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        end = end or date.today()
        ticker = ticker.upper()

        if use_cache:
            cached = cache.load(ticker)
            if cached is not None and cache.is_fresh(ticker) and cache.covers(cached, start):
                sliced = cached.loc[(cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))]
                if not sliced.empty:
                    return sliced, {"cache": "hit (cached earlier today, no API calls made)"}

        frames: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}

        for source in self.sources:
            if not source.is_configured():
                errors[source.name] = "not configured (missing API key)"
                continue
            try:
                frames[source.name] = source.fetch_ohlcv(ticker, start, end)
            except DataSourceUnavailable as exc:
                errors[source.name] = str(exc)
                logger.warning("%s unavailable for %s: %s", source.name, ticker, exc)
            except Exception as exc:  # noqa: BLE001 - a flaky provider must not break the others
                errors[source.name] = f"unexpected error: {exc}"
                logger.warning("%s failed for %s: %s", source.name, ticker, exc)

        if not frames:
            raise DataSourceUnavailable(
                f"No data source returned results for '{ticker}'. Errors: {errors}"
            )

        combined = self._combine(frames)
        if use_cache:
            cache.save(ticker, combined)

        status = {name: "ok" for name in frames} | errors
        return combined, status

    @staticmethod
    def _combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(frames) == 1:
            return next(iter(frames.values())).sort_index()

        panel = pd.concat(frames.values(), keys=frames.keys(), axis=1)  # MultiIndex columns (source, field)
        combined = pd.DataFrame(index=panel.index)
        for field in OHLCV_COLUMNS:
            combined[field] = panel.xs(field, axis=1, level=1).median(axis=1, skipna=True)
        return combined.dropna(how="any").sort_index()


def default_lookback_start(years: int) -> date:
    return date.today() - timedelta(days=365 * years + 30)
