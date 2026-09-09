from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.config import DATA_CACHE_DIR


def _cache_path(ticker: str) -> Path:
    return DATA_CACHE_DIR / f"{ticker.upper()}.csv"


def load(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def save(ticker: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(ticker), index_label="date")


def is_fresh(ticker: str) -> bool:
    """Daily-bar data only changes once a session closes, so a cache written
    earlier today is still correct — no need to re-hit rate-limited APIs again
    until tomorrow."""
    path = _cache_path(ticker)
    if not path.exists():
        return False
    return datetime.fromtimestamp(path.stat().st_mtime).date() == date.today()


_COVERAGE_TOLERANCE_DAYS = 60  # providers' historical limits rarely land on an exact date


def covers(df: pd.DataFrame, start: date) -> bool:
    """True if the cache reaches back to (roughly) `start`.

    Allows slack because `start` is typically padded by default_lookback_start()
    to account for non-trading days, and free-tier providers often cap history at
    "about N years" rather than an exact date — so the cache's earliest date can
    land a bit later than the strict request without meaning the cache is stale.
    """
    if df.empty:
        return False
    return df.index.min().date() <= start + timedelta(days=_COVERAGE_TOLERANCE_DAYS)
