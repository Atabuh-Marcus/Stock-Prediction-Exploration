from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from app.features.indicators import atr as atr_indicator

# Retail risk-management convention: risk ~1% of account equity per trade. The app
# has no notion of the user's actual account size, so position sizing comes out as
# a % of portfolio rather than a dollar amount or share count.
RISK_PER_TRADE_PCT = 1.0
MAX_POSITION_PCT = 15.0
STOP_ATR_MULTIPLE = 1.5
REWARD_RISK_RATIO = 1.5  # take-profit distance = this many times the stop distance

# ML confidence is the only piece that's actually backtested and calibrated
# (classification_brier_score) — it gets the larger share of the composite score;
# the indicator breakdown is a secondary confirmation/dissent check.
ML_WEIGHT = 0.6
INDICATOR_WEIGHT = 0.4


@dataclass
class IndicatorVote:
    name: str
    verdict: str  # "bullish" | "bearish" | "neutral"
    detail: str


@dataclass
class TradingSignal:
    rating: str  # "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"
    composite_score: float  # -1..1
    bullish_count: int
    bearish_count: int
    neutral_count: int
    indicators: list[IndicatorVote]
    trade_direction: str | None  # "long" | "short" | None (Hold)
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    risk_reward_ratio: float | None
    atr: float
    atr_pct: float
    suggested_position_pct: float
    position_sizing_note: str
    note: str


def _indicator_votes(latest: pd.Series, ohlcv: pd.DataFrame) -> list[IndicatorVote]:
    votes: list[IndicatorVote] = []

    rsi = latest.get("rsi_14d")
    if rsi is not None and pd.notna(rsi):
        if rsi < 30:
            votes.append(IndicatorVote("RSI (14d)", "bullish", f"{rsi:.1f} — oversold"))
        elif rsi > 70:
            votes.append(IndicatorVote("RSI (14d)", "bearish", f"{rsi:.1f} — overbought"))
        else:
            votes.append(IndicatorVote("RSI (14d)", "neutral", f"{rsi:.1f} — neutral range"))

    macd_hist = latest.get("macd_hist")
    if macd_hist is not None and pd.notna(macd_hist):
        if macd_hist > 0.0005:
            votes.append(IndicatorVote("MACD", "bullish", "histogram above signal line"))
        elif macd_hist < -0.0005:
            votes.append(IndicatorVote("MACD", "bearish", "histogram below signal line"))
        else:
            votes.append(IndicatorVote("MACD", "neutral", "tracking the signal line"))

    sma_gap = latest.get("sma_gap_20d")
    if sma_gap is not None and pd.notna(sma_gap):
        if sma_gap > 0.01:
            votes.append(IndicatorVote("Trend (vs. 20d SMA)", "bullish", f"{sma_gap * 100:+.1f}% above"))
        elif sma_gap < -0.01:
            votes.append(IndicatorVote("Trend (vs. 20d SMA)", "bearish", f"{sma_gap * 100:+.1f}% below"))
        else:
            votes.append(IndicatorVote("Trend (vs. 20d SMA)", "neutral", "tracking the 20d average"))

    close = ohlcv["close"]
    if len(close) >= 20:
        mid = close.rolling(20).mean().iloc[-1]
        std = close.rolling(20).std().iloc[-1]
        if pd.notna(mid) and pd.notna(std) and std > 0:
            upper, lower = mid + 2 * std, mid - 2 * std
            last = close.iloc[-1]
            if last <= lower:
                votes.append(IndicatorVote("Bollinger Bands", "bullish", "price at/below the lower band"))
            elif last >= upper:
                votes.append(IndicatorVote("Bollinger Bands", "bearish", "price at/above the upper band"))
            else:
                votes.append(IndicatorVote("Bollinger Bands", "neutral", "inside the bands"))

    rel_5d = latest.get("spy_relative_return_5d")
    if rel_5d is not None and pd.notna(rel_5d):
        if rel_5d > 0.005:
            votes.append(IndicatorVote("Relative Strength (vs S&P 500, 5d)", "bullish", f"{rel_5d * 100:+.2f}% excess return"))
        elif rel_5d < -0.005:
            votes.append(IndicatorVote("Relative Strength (vs S&P 500, 5d)", "bearish", f"{rel_5d * 100:+.2f}% excess return"))
        else:
            votes.append(IndicatorVote("Relative Strength (vs S&P 500, 5d)", "neutral", "tracking the market"))

    sentiment = latest.get("news_sentiment_5d_avg")
    if sentiment is not None and pd.notna(sentiment):
        if sentiment > 0.05:
            votes.append(IndicatorVote("News Sentiment (5d avg)", "bullish", f"score {sentiment:+.3f}"))
        elif sentiment < -0.05:
            votes.append(IndicatorVote("News Sentiment (5d avg)", "bearish", f"score {sentiment:+.3f}"))
        else:
            votes.append(IndicatorVote("News Sentiment (5d avg)", "neutral", f"score {sentiment:+.3f}"))

    return votes


def _composite_score(direction: str, confidence: float, votes: list[IndicatorVote]) -> float:
    ml_signed = confidence if direction == "rise" else -confidence
    if votes:
        bullish = sum(1 for v in votes if v.verdict == "bullish")
        bearish = sum(1 for v in votes if v.verdict == "bearish")
        indicator_signed = (bullish - bearish) / len(votes)
    else:
        indicator_signed = 0.0
    return ML_WEIGHT * ml_signed + INDICATOR_WEIGHT * indicator_signed


def _rating_from_score(score: float) -> str:
    if score >= 0.5:
        return "Strong Buy"
    if score >= 0.15:
        return "Buy"
    if score <= -0.5:
        return "Strong Sell"
    if score <= -0.15:
        return "Sell"
    return "Hold"


def _risk_levels(entry: float, atr_value: float, trade_direction: str | None) -> tuple[float | None, float | None]:
    if trade_direction is None or atr_value <= 0 or entry <= 0:
        return None, None
    stop_distance = STOP_ATR_MULTIPLE * atr_value
    if trade_direction == "long":
        return entry - stop_distance, entry + stop_distance * REWARD_RISK_RATIO
    return entry + stop_distance, entry - stop_distance * REWARD_RISK_RATIO


def _position_sizing(entry: float, stop_loss: float | None, confidence: float) -> tuple[float, str]:
    if stop_loss is None or entry <= 0:
        return 0.0, "Hold rating — no qualifying trade setup, so no position is sized."

    stop_distance_fraction = abs(entry - stop_loss) / entry
    if stop_distance_fraction <= 0:
        return 0.0, "Stop distance too small to size a position safely."

    risk_fraction = RISK_PER_TRADE_PCT / 100
    raw_position_pct = (risk_fraction / stop_distance_fraction) * 100
    # Scale toward zero as confidence approaches a coin flip (50%); full size by ~85%+.
    confidence_factor = min(1.0, max(0.0, (confidence - 0.5) / 0.35))
    sized_pct = min(raw_position_pct * confidence_factor, MAX_POSITION_PCT)

    note = (
        f"Assumes risking {RISK_PER_TRADE_PCT:.0f}% of account equity on this trade (a standard "
        f"retail default, not a fact about your actual portfolio), scaled down for confidence, "
        f"capped at {MAX_POSITION_PCT:.0f}% of a single position."
    )
    return round(sized_pct, 2), note


def compute_trading_signal(
    ohlcv: pd.DataFrame, latest_features: pd.Series, direction: str, confidence: float
) -> TradingSignal:
    """Turns the raw direction/confidence prediction into an actionable signal:
    a composite Buy/Sell rating (ML confidence + indicator agreement), ATR-based
    stop-loss/take-profit levels, and a volatility- and confidence-adjusted
    suggested position size."""
    entry = float(ohlcv["close"].iloc[-1])
    atr_series = atr_indicator(ohlcv["high"], ohlcv["low"], ohlcv["close"], window=14)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    atr_pct = (atr_value / entry * 100) if entry else 0.0

    votes = _indicator_votes(latest_features, ohlcv)
    score = _composite_score(direction, confidence, votes)
    rating = _rating_from_score(score)

    trade_direction = None
    if rating in ("Buy", "Strong Buy"):
        trade_direction = "long"
    elif rating in ("Sell", "Strong Sell"):
        trade_direction = "short"

    stop_loss, take_profit = _risk_levels(entry, atr_value, trade_direction)
    position_pct, sizing_note = _position_sizing(entry, stop_loss, confidence)

    bullish = sum(1 for v in votes if v.verdict == "bullish")
    bearish = sum(1 for v in votes if v.verdict == "bearish")
    neutral = len(votes) - bullish - bearish

    if trade_direction == "short":
        note = (
            "The backtest (see /backtest) only simulates a long/cash strategy — this short setup "
            "is a symmetric extrapolation from the same signals, not separately backtested."
        )
    else:
        note = "Composite of model confidence and indicator agreement. Informational only — not financial advice."

    return TradingSignal(
        rating=rating,
        composite_score=round(score, 4),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        indicators=votes,
        trade_direction=trade_direction,
        entry_price=round(entry, 4),
        stop_loss=round(stop_loss, 4) if stop_loss is not None else None,
        take_profit=round(take_profit, 4) if take_profit is not None else None,
        risk_reward_ratio=REWARD_RISK_RATIO if stop_loss is not None else None,
        atr=round(atr_value, 4),
        atr_pct=round(atr_pct, 3),
        suggested_position_pct=position_pct,
        position_sizing_note=sizing_note,
        note=note,
    )


def to_dict(signal: TradingSignal) -> dict:
    return asdict(signal)
