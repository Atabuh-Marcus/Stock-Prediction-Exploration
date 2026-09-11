from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: str  # "stocks" | "crypto" | "forex" | "indices" | "commodities"


# A curated, diversified instrument list rather than a literal "every symbol that
# exists" — no data provider here offers a discovery/listing endpoint, and an
# unbounded universe would blow through Alpha Vantage's 25-requests/day free quota
# and make regular retraining impractically slow. Sized to stay well within that
# while still covering every asset class the app supports.
MARKET_UNIVERSE: list[Instrument] = [
    # Stocks — mega-cap, cross-sector
    Instrument("AAPL", "Apple", "stocks"),
    Instrument("MSFT", "Microsoft", "stocks"),
    Instrument("GOOGL", "Alphabet", "stocks"),
    Instrument("AMZN", "Amazon", "stocks"),
    Instrument("NVDA", "Nvidia", "stocks"),
    Instrument("META", "Meta Platforms", "stocks"),
    Instrument("TSLA", "Tesla", "stocks"),
    Instrument("JPM", "JPMorgan Chase", "stocks"),
    Instrument("JNJ", "Johnson & Johnson", "stocks"),
    Instrument("WMT", "Walmart", "stocks"),
    # Crypto
    Instrument("BTC-USD", "Bitcoin", "crypto"),
    Instrument("ETH-USD", "Ethereum", "crypto"),
    Instrument("SOL-USD", "Solana", "crypto"),
    # Forex
    Instrument("EURUSD=X", "EUR / USD", "forex"),
    Instrument("GBPUSD=X", "GBP / USD", "forex"),
    Instrument("USDJPY=X", "USD / JPY", "forex"),
    # Indices
    Instrument("^GSPC", "S&P 500", "indices"),
    Instrument("^DJI", "Dow Jones Industrial Average", "indices"),
    Instrument("^IXIC", "Nasdaq Composite", "indices"),
    # Commodities (futures)
    Instrument("GC=F", "Gold", "commodities"),
    Instrument("CL=F", "Crude Oil (WTI)", "commodities"),
    Instrument("SI=F", "Silver", "commodities"),
]

ASSET_CLASS_LABELS = {
    "stocks": "Stocks",
    "crypto": "Crypto",
    "forex": "Forex",
    "indices": "Indices",
    "commodities": "Commodities",
}

MARKET_UNIVERSE_SYMBOLS = [i.symbol for i in MARKET_UNIVERSE]
