from __future__ import annotations

from datetime import date

import pandas as pd

from app.config import PREDICTION_LOG_DIR

LOG_PATH = PREDICTION_LOG_DIR / "log.csv"

COLUMNS = [
    "ticker",
    "as_of_date",
    "horizon_days",
    "predicted_direction",
    "confidence",
    "predicted_price",
    "last_close",
    "target_date",
    "resolved",
    "actual_close",
    "actual_direction",
    "correct",
    "realized_change_pct",
]

CONFIDENCE_BUCKET_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
CONFIDENCE_BUCKET_LABELS = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]


def _load() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH, parse_dates=["as_of_date", "target_date"])


def _save(df: pd.DataFrame) -> None:
    df.to_csv(LOG_PATH, index=False)


def record_prediction(
    ticker: str,
    as_of_date: str,
    horizon_days: int,
    predicted_direction: str,
    confidence: float,
    predicted_price: float,
    last_close: float,
) -> None:
    """Logs a prediction, deduped by (ticker, as_of_date) — repeat predict calls
    for the same ticker on the same trading day reuse the same cached data and
    would produce the same prediction anyway, so they don't create duplicate rows.
    """
    df = _load()
    as_of_ts = pd.Timestamp(as_of_date)
    if ((df["ticker"] == ticker) & (df["as_of_date"] == as_of_ts)).any():
        return

    target_date = as_of_ts + pd.tseries.offsets.BDay(horizon_days)
    row = {
        "ticker": ticker,
        "as_of_date": as_of_ts,
        "horizon_days": horizon_days,
        "predicted_direction": predicted_direction,
        "confidence": confidence,
        "predicted_price": predicted_price,
        "last_close": last_close,
        "target_date": target_date,
        "resolved": False,
        "actual_close": None,
        "actual_direction": None,
        "correct": None,
        "realized_change_pct": None,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)


def reconcile(ticker: str | None = None) -> None:
    """Fills in actual outcomes for past predictions whose target date has
    arrived, reusing the same data pipeline predict_ticker already uses.
    Best-effort: a ticker whose data can't be fetched right now just stays
    unresolved and gets picked up on a later call.
    """
    df = _load()
    if df.empty:
        return

    unresolved_mask = ~df["resolved"].astype(bool)
    if ticker:
        unresolved_mask &= df["ticker"] == ticker
    if not unresolved_mask.any():
        return

    from app.data.aggregator import DataAggregator  # local import avoids a data<->models import cycle

    today = pd.Timestamp(date.today())
    for tkr in df.loc[unresolved_mask, "ticker"].unique():
        due_mask = unresolved_mask & (df["ticker"] == tkr) & (df["target_date"] <= today)
        if not due_mask.any():
            continue
        try:
            earliest_target = df.loc[due_mask, "target_date"].min().date()
            ohlcv, _ = DataAggregator().fetch(tkr, start=earliest_target)
        except Exception:
            continue

        for idx in df.index[due_mask]:
            row = df.loc[idx]
            future = ohlcv[ohlcv.index >= row["target_date"]]
            if future.empty:
                continue  # target date hasn't traded yet in available data
            actual_close = float(future["close"].iloc[0])
            actual_direction = "rise" if actual_close > row["last_close"] else "fall"
            df.loc[idx, "actual_close"] = actual_close
            df.loc[idx, "actual_direction"] = actual_direction
            df.loc[idx, "correct"] = actual_direction == row["predicted_direction"]
            df.loc[idx, "realized_change_pct"] = (actual_close / row["last_close"] - 1) * 100
            df.loc[idx, "resolved"] = True

    _save(df)


def _calibration_buckets(resolved: pd.DataFrame) -> list[dict]:
    """Buckets resolved predictions by confidence and reports actual accuracy per
    bucket — the real-world check of whether "70% confident" predictions are
    actually right about 70% of the time, using live outcomes rather than the
    backtest's simulated ones."""
    if resolved.empty:
        return []
    conf = resolved["confidence"].astype(float)
    correct = resolved["correct"].astype(bool)
    buckets = []
    for i, label in enumerate(CONFIDENCE_BUCKET_LABELS):
        mask = (conf >= CONFIDENCE_BUCKET_EDGES[i]) & (conf < CONFIDENCE_BUCKET_EDGES[i + 1])
        if mask.sum() == 0:
            continue
        buckets.append({"range": label, "count": int(mask.sum()), "actual_accuracy": round(float(correct[mask].mean()), 4)})
    return buckets


def get_history(ticker: str | None = None) -> dict:
    reconcile(ticker)
    df = _load()
    if ticker:
        df = df[df["ticker"] == ticker.upper().strip()]
    df = df.sort_values("as_of_date", ascending=False)

    resolved = df[df["resolved"].astype(bool)]
    accuracy = float(resolved["correct"].astype(bool).mean()) if not resolved.empty else None

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "ticker": r["ticker"],
                "as_of_date": str(pd.Timestamp(r["as_of_date"]).date()),
                "target_date": str(pd.Timestamp(r["target_date"]).date()),
                "horizon_days": int(r["horizon_days"]),
                "predicted_direction": r["predicted_direction"],
                "confidence": float(r["confidence"]),
                "predicted_price": float(r["predicted_price"]),
                "last_close": float(r["last_close"]),
                "resolved": bool(r["resolved"]),
                "actual_close": None if pd.isna(r["actual_close"]) else float(r["actual_close"]),
                "actual_direction": None if pd.isna(r["actual_direction"]) else r["actual_direction"],
                "correct": None if pd.isna(r["correct"]) else bool(r["correct"]),
                "realized_change_pct": None if pd.isna(r["realized_change_pct"]) else float(r["realized_change_pct"]),
            }
        )

    return {
        "total_predictions": len(df),
        "resolved_predictions": len(resolved),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "calibration_buckets": _calibration_buckets(resolved),
        "rows": rows,
    }
