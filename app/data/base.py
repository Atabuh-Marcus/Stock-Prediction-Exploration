from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSourceUnavailable(RuntimeError):
    """Raised when a source is not configured (missing API key) or fails to return data."""


class DataSource(ABC):
    """Common interface every market data provider implements.

    fetch_ohlcv must return a DataFrame indexed by date (ascending, no duplicates)
    with exactly the OHLCV_COLUMNS columns, in the source's native currency/units.
    """

    name: str

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...
