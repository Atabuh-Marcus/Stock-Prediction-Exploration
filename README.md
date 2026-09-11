# Stock Prediction

Predicts a stock's next-session **direction** (rise/fall, with confidence) and **price** from
technical indicators, using data combined across multiple market data providers.

Not financial advice — this is a technical-indicator model trained on historical prices, which is
a fundamentally noisy prediction problem. Treat outputs as one input among many, not a signal to trade on.

## How it works

- **Data (`app/data/`)**: pulls OHLCV from [yfinance](https://github.com/ranaroussi/yfinance) (free,
  no key, always on), [Alpha Vantage](https://www.alphavantage.co/) and [Polygon.io](https://polygon.io/)
  (both optional — used automatically if you set an API key, skipped otherwise). The aggregator
  outer-joins whatever sources responded and takes the per-day **median** across them, so one
  source glitching on a given day doesn't skew the combined series. Combined results are cached to
  `data_cache/<TICKER>.csv` for the rest of the day, so repeated predict/train/backtest calls don't
  keep re-hitting rate-limited free-tier APIs — pass `refresh=True` / `--refresh` to force a re-fetch.
- **Features (`app/features/`)**: returns, SMA/EMA gaps, RSI, MACD (normalized by price), Bollinger
  bandwidth, rolling volatility, and volume z-score, computed from the combined series — plus two
  feature groups that need outside context a single ticker's own price history can't provide:
  - **Market context**: the stock's return relative to the S&P 500 (SPY) at 1/5/10 days, and SPY's
    own rolling volatility as a market-wide risk proxy. A stock's move *relative to the market* is
    typically far more informative than its price action in isolation.
  - **News sentiment**: daily-aggregated sentiment from Alpha Vantage's `NEWS_SENTIMENT` endpoint
    (forward-filled between news events, smoothed with a 5-day average). Only active when
    `ALPHAVANTAGE_API_KEY` is set — neutral/zero otherwise, so the model degrades gracefully rather
    than breaking.
- **Models (`app/models/`)**: a `HistGradientBoostingClassifier` (direction) and
  `HistGradientBoostingRegressor` (next-period **return**, reconstructed into a price as
  `last_close * (1 + predicted_return)` — this keeps the price prediction anchored to where the
  stock actually is, rather than an absolute price a mostly scale-invariant feature set can't anchor)
  per ticker, trained on a time-based split (no shuffling) and saved to `models_store/<TICKER>.joblib`.
  - **Hyperparameter tuning** (`app/models/model_factory.py`): `RandomizedSearchCV` with
    `TimeSeriesSplit` cross-validation (folds respect chronological order — no future data ever
    leaks into a training fold) searches tree depth, learning rate, regularization, and leaf size for
    both models.
  - **Confidence calibration**: the classifier is wrapped in `CalibratedClassifierCV` (Platt/sigmoid
    scaling), so a reported "70% confidence" is actually checked against how often the model is right
    at that confidence level, rather than being an arbitrary raw score. `classification_brier_score`
    in the metrics (lower is better; 0.25 = a coin flip) is the metric that actually measures this —
    plain accuracy doesn't.
- **Backtesting (`app/models/backtest.py`)**: walk-forward validation — retrains periodically using
  only data available up to that point, then simulates a simple long/cash strategy (long when
  predicted direction is "rise" with confidence over a threshold, flat otherwise) against a buy & hold
  benchmark. Reports total return, Sharpe, max drawdown, and out-of-sample directional hit rate.
- **Interfaces**: a CLI, a FastAPI backend, a browser frontend, and a Jupyter notebook — all built
  on the same `app/data`, `app/features`, `app/models` modules, so results are consistent across all four.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optionally add ALPHAVANTAGE_API_KEY / POLYGON_API_KEY
```

yfinance works with no key. Add the others in `.env` to have their data folded into the combined series.

## Usage

### CLI

```bash
python -m app.cli train AAPL                       # train (or retrain) a model
python -m app.cli predict AAPL                      # predict direction + price (auto-trains if no model saved yet)
python -m app.cli watchlist AAPL MSFT GOOGL         # predict several tickers at once, ranked by confidence
python -m app.cli backtest AAPL                     # walk-forward backtest vs buy & hold
python -m app.cli serve                             # run the API + web app on http://127.0.0.1:8000
```

`train`, `predict`, and `backtest` all accept `--refresh` to bypass the data cache.

### Web app

```bash
python -m app.cli serve --reload
```

Then open http://127.0.0.1:8000 — three tabs:

- **Predict** — single-ticker direction/price prediction with a price chart.
- **Watchlist** — add tickers (persisted in your browser via localStorage), run them all at once,
  see a table ranked by confidence.
- **Backtest** — walk-forward strategy performance vs buy & hold, with an equity-curve chart.

First prediction/backtest for a ticker trains a model (takes a few seconds to a minute); later runs
reuse the saved model until you hit Retrain.

### API

- `GET /predict?symbol=AAPL` — direction, confidence, predicted price, model metrics
- `GET /watchlist?symbols=AAPL,MSFT,GOOGL` — predicts each ticker independently (max 20; one bad
  ticker doesn't fail the batch)
- `GET /backtest?symbol=AAPL&lookback_years=5&retrain_every_days=20&confidence_threshold=0.5` —
  walk-forward backtest, returns metrics + an equity curve
- `POST /train?symbol=AAPL&lookback_years=5&horizon_days=1` — (re)trains and saves the model
- `GET /history?symbol=AAPL&days=180` — combined OHLCV history for charting
- `GET /health`

All of `/predict`, `/train`, and `/backtest` accept `refresh=true` to bypass the data cache.

### Notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

Walks through fetching combined data, inspecting indicators, training, and checking feature
importance interactively.

## Configuration

Env vars (set in `.env`, see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ALPHAVANTAGE_API_KEY` | unset | enables Alpha Vantage as a second data source |
| `POLYGON_API_KEY` | unset | enables Polygon.io as a third data source |
| `PREDICTION_HORIZON_DAYS` | `1` | how many trading days ahead to predict |
| `DEFAULT_LOOKBACK_YEARS` | `5` | history pulled when training from scratch |

## Project layout

```
app/
  config.py            env vars, paths
  data/                per-provider fetchers, aggregator (combine), cache (disk cache)
  features/            technical indicators + labeled feature matrix
  models/               train.py (fit + save), predict.py (load + infer, watchlist), backtest.py
  cli.py                 train / predict / watchlist / backtest / serve subcommands
  main.py                FastAPI app (serves the API and the web/ frontend)
web/                     browser frontend (Predict / Watchlist / Backtest tabs)
notebooks/               interactive exploration
models_store/            saved model bundles (*.joblib, gitignored)
data_cache/               per-ticker OHLCV cache, refreshed daily (gitignored)
```
