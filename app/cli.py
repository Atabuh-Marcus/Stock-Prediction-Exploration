from __future__ import annotations

import argparse
import json
import sys

from app.config import DEFAULT_LOOKBACK_YEARS, PREDICTION_HORIZON_DAYS


def cmd_train(args: argparse.Namespace) -> None:
    from app.models.train import train_ticker

    bundle = train_ticker(
        args.ticker, lookback_years=args.lookback_years, horizon_days=args.horizon_days, refresh=args.refresh
    )
    print(f"Trained model for {bundle.ticker}")
    print(f"  Trained at:        {bundle.trained_at}")
    print(f"  Data sources used: {bundle.source_status}")
    print(f"  Metrics:           {json.dumps(bundle.metrics, indent=2)}")


def cmd_predict(args: argparse.Namespace) -> None:
    from app.models.predict import predict_ticker

    result = predict_ticker(args.ticker, refresh=args.refresh)
    _print_prediction(result)


def cmd_watchlist(args: argparse.Namespace) -> None:
    from app.models.predict import predict_watchlist

    entries = predict_watchlist(args.tickers)
    print(f"\n{'TICKER':<8}{'DIRECTION':<12}{'CONFIDENCE':<13}{'LAST CLOSE':<13}{'PREDICTED':<13}{'CHANGE':<10}")
    print("-" * 69)
    for entry in entries:
        if entry.error:
            print(f"{entry.ticker:<8}ERROR: {entry.error}")
            continue
        r = entry.prediction
        arrow = "▲" if r.direction == "rise" else "▼"
        print(
            f"{r.ticker:<8}{arrow} {r.direction.upper():<10}{r.direction_confidence:<13.1%}"
            f"${r.last_close:<12.2f}${r.predicted_price:<12.2f}{r.predicted_change_pct:+.2f}%"
        )
    print("\nInformational only — not financial advice.\n")


def cmd_backtest(args: argparse.Namespace) -> None:
    from app.models.backtest import run_backtest

    result = run_backtest(
        args.ticker,
        lookback_years=args.lookback_years,
        horizon_days=args.horizon_days,
        retrain_every_days=args.retrain_every_days,
        confidence_threshold=args.confidence_threshold,
        refresh=args.refresh,
    )
    print(f"\nBacktest: {result.ticker}  ({result.start_date} to {result.end_date}, {result.evaluated_days} days)")
    print(f"  Retrains (walk-forward): {result.retrains}")
    print(f"  Directional hit rate:    {result.hit_rate:.1%}")
    print(f"  Days in market:          {result.days_in_market_pct:.1f}%")
    print(f"  Strategy total return:   {result.strategy_total_return_pct:+.2f}%")
    print(f"  Buy & hold total return: {result.buy_hold_total_return_pct:+.2f}%")
    print(f"  Strategy Sharpe:         {result.strategy_sharpe:.2f}")
    print(f"  Strategy max drawdown:   {result.strategy_max_drawdown_pct:.2f}%")
    print(f"\n  {result.note}\n")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def _print_prediction(result) -> None:
    arrow = "▲" if result.direction == "rise" else "▼"
    print(f"\n{result.ticker}  (as of {result.as_of_date}, {result.horizon_days}-day horizon)")
    print(f"  Last close:        ${result.last_close}")
    print(f"  Direction:         {arrow} {result.direction.upper()}  ({result.direction_confidence:.1%} confidence)")
    print(f"  Predicted price:   ${result.predicted_price}  ({result.predicted_change_pct:+.2f}%)")
    print(f"  Data sources:      {result.data_sources}")
    print(f"  Model metrics:     {json.dumps(result.model_metrics, indent=2)}")
    print("\n  Informational only — not financial advice.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockpred", description="Stock rise/fall prediction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train (or retrain) a model for a ticker")
    train_parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    train_parser.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK_YEARS)
    train_parser.add_argument("--horizon-days", type=int, default=PREDICTION_HORIZON_DAYS)
    train_parser.add_argument("--refresh", action="store_true", help="Bypass the data cache, force a fresh fetch")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Predict rise/fall + price for a ticker")
    predict_parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    predict_parser.add_argument("--refresh", action="store_true", help="Bypass the data cache, force a fresh fetch")
    predict_parser.set_defaults(func=cmd_predict)

    watchlist_parser = subparsers.add_parser("watchlist", help="Predict several tickers at once, ranked by confidence")
    watchlist_parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT GOOGL")
    watchlist_parser.set_defaults(func=cmd_watchlist)

    backtest_parser = subparsers.add_parser("backtest", help="Walk-forward backtest a ticker's strategy performance")
    backtest_parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    backtest_parser.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK_YEARS)
    backtest_parser.add_argument("--horizon-days", type=int, default=PREDICTION_HORIZON_DAYS)
    backtest_parser.add_argument("--retrain-every-days", type=int, default=20)
    backtest_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    backtest_parser.add_argument("--refresh", action="store_true", help="Bypass the data cache, force a fresh fetch")
    backtest_parser.set_defaults(func=cmd_backtest)

    serve_parser = subparsers.add_parser("serve", help="Run the API + web app")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI top-level error boundary
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
