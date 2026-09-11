from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from app.config import ALPHA_VANTAGE_API_KEY
from app.data import cache

logger = logging.getLogger(__name__)

API_URL = "https://www.alphavantage.co/query"
MAX_ARTICLES_PER_CALL = 1000  # Alpha Vantage's hard cap on NEWS_SENTIMENT's `limit` param


def is_configured() -> bool:
    return bool(ALPHA_VANTAGE_API_KEY)


def fetch_daily_sentiment(ticker: str, start: date, use_cache: bool = True) -> pd.DataFrame | None:
    """Daily-aggregated news sentiment for `ticker` from Alpha Vantage's
    NEWS_SENTIMENT endpoint. Returns a DataFrame indexed by date with
    `sentiment_score` (mean of that day's article-level ticker sentiment,
    range roughly -1..1) and `article_count`. None if unavailable — this is an
    optional enhancement, not a hard requirement, so callers should degrade
    gracefully (treat as neutral/no sentiment data) rather than fail.

    Uses the same daily on-disk cache as OHLCV data (kind="sentiment") since
    this shares the same 25-requests/day free-tier budget and only needs
    refreshing once per day.
    """
    ticker = ticker.upper().strip()

    if use_cache:
        cached = cache.load(ticker, kind="sentiment")
        if cached is not None and cache.is_fresh(ticker, kind="sentiment") and cache.covers(cached, start):
            return cached

    if not is_configured():
        return None

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": f"{start.strftime('%Y%m%d')}T0000",
        "limit": MAX_ARTICLES_PER_CALL,
        "sort": "EARLIEST",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - sentiment is optional, never break the pipeline over it
        logger.warning("News sentiment fetch failed for %s: %s", ticker, exc)
        return None

    if "Note" in payload or "Information" in payload:
        logger.warning(
            "Alpha Vantage news sentiment rate-limited/restricted for %s: %s",
            ticker,
            payload.get("Note") or payload.get("Information"),
        )
        return None

    feed = payload.get("feed")
    if not feed:
        return None

    rows = []
    for article in feed:
        published = article.get("time_published")  # format: YYYYMMDDTHHMMSS
        if not published:
            continue
        ticker_scores = {
            t["ticker"]: float(t["ticker_sentiment_score"])
            for t in article.get("ticker_sentiment", [])
            if "ticker" in t and "ticker_sentiment_score" in t
        }
        score = ticker_scores.get(ticker)
        if score is None:
            continue
        rows.append({"date": pd.Timestamp(published[:8]), "sentiment_score": score})

    if not rows:
        return None

    articles = pd.DataFrame(rows)
    daily = articles.groupby("date").agg(
        sentiment_score=("sentiment_score", "mean"), article_count=("sentiment_score", "size")
    )
    daily.index.name = "date"
    daily = daily.sort_index()

    if use_cache:
        cache.save(ticker, daily, kind="sentiment")

    return daily
