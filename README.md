# Stock Prediction

Predicts an instrument's next-session **direction** (rise/fall, with confidence) and **price** from
technical indicators, using data combined across multiple market data providers. Works on any symbol
yfinance recognizes, not just stocks — see **Asset classes** below.

Not financial advice — this is a technical-indicator model trained on historical prices, which is
a fundamentally noisy prediction problem. Treat outputs as one input among many, not a signal to trade on.

## Asset classes

Every ticker field accepts plain equities and yfinance-style symbols for other asset classes — the
data layer, features, and trading signal all work the same way regardless of asset class:

| Asset class | Example symbols |
|---|---|
| Stocks | `AAPL`, `MSFT`, `TSLA` |
| Crypto | `BTC-USD`, `ETH-USD` |
| Forex | `EURUSD=X`, `GBPJPY=X` |
| Indices | `^GSPC` (S&P 500), `^DJI`, `^IXIC` |
| Commodities (futures) | `GC=F` (gold), `CL=F` (crude oil) |

Alpha Vantage and Polygon are stock-focused, so for non-equity symbols the combined series usually
falls back to yfinance alone (still fine — see `data_sources` in the response). Spot forex pairs
report zero volume from Yahoo (no central exchange tape); the volume-derived features are neutralized
to zero in that case instead of breaking the pipeline, rather than treated as missing data.

## Market universe

`app/universe.py` defines a curated list of ~20 instruments spanning every asset class above (mega-cap
stocks, top cryptos, major forex pairs, key indices and commodities) — not a literal "every symbol
that exists" (no provider here offers a discovery/listing endpoint), sized to stay within Alpha
Vantage's 25-requests/day free quota and keep regular retraining practical. It's the single source of
truth for:

- The web app's **Markets** tab (see below).
- `stockpred watchlist` with no arguments, and the daily automation script.
- `stockpred train-all`, and the weekly retraining automation.

Edit `MARKET_UNIVERSE` in that file to track a different set.

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
- **Trading signals (`app/models/trading_signals.py`)**: turns the raw direction/confidence output into
  something actionable —
  - A **composite Buy/Sell/Hold rating** (Strong Buy → Strong Sell) blending the ML confidence (60%,
    weighted higher since it's the only piece that's actually backtested/calibrated) with agreement
    across six independent technical indicators (RSI, MACD, trend vs. 20d SMA, Bollinger Bands,
    relative strength vs. SPY, news sentiment) — 40%.
  - An **indicator agreement breakdown** showing which indicators are bullish, bearish, or neutral and
    why, so the rating isn't a black box.
  - **ATR-based stop-loss / take-profit levels** on a Buy/Sell rating (1.5× ATR stop, 1.5:1 reward:risk),
    with a `Hold` giving no trade setup.
  - A **volatility- and confidence-adjusted suggested position size** (% of portfolio), from a standard
    1%-of-equity risk-per-trade assumption scaled down as confidence approaches a coin flip, capped at
    15% of a single position.
- **Backtesting (`app/models/backtest.py`)**: walk-forward validation — retrains periodically using
  only data available up to that point, then simulates a simple long/cash strategy (long when
  predicted direction is "rise" with confidence over a threshold, flat otherwise) against a buy & hold
  benchmark. Reports total return, Sharpe, max drawdown, and out-of-sample directional hit rate.
- **Prediction history (`app/models/prediction_log.py`)**: every prediction (from any interface) is
  logged to `prediction_log/log.csv`, deduped per ticker/day. Once a prediction's target date has
  passed, it's automatically reconciled against the actual close price — so over time you get a real
  track record, not just a backtest's simulated one, plus a live calibration check (accuracy broken
  down by confidence bucket).
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
python -m app.cli watchlist                         # same, but for the full curated market universe
python -m app.cli train-all                         # retrain every instrument in the market universe
python -m app.cli backtest AAPL                     # walk-forward backtest vs buy & hold
python -m app.cli history [TICKER]                  # past predictions + how they resolved, calibration by confidence bucket
python -m app.cli serve                             # run the API + web app on http://127.0.0.1:8000
```

`train`, `predict`, `backtest`, and `train-all` all accept `--refresh` to bypass the data cache.

### Web app

```bash
python -m app.cli serve --reload
```

Then open http://127.0.0.1:8000 — five tabs:

- **Predict** — single-ticker direction/price prediction, a Trading Signal card (Buy/Sell/Hold rating,
  indicator breakdown, stop-loss/take-profit, suggested position size), a price chart, and a Signals
  panel (market context, news sentiment, calibration quality) so what's driving the prediction is
  visible instead of buried in a JSON blob.
- **Watchlist** — add tickers (persisted in your browser via localStorage), run them all at once, see
  a table ranked by confidence with each ticker's rating, plus a normalized performance-comparison
  chart across the same tickers.
- **Backtest** — walk-forward strategy performance vs buy & hold, with an equity-curve chart.
- **History** — every prediction ever made, with resolved outcomes and a real calibration check
  (accuracy by confidence bucket) as they accumulate over time.
- **Markets** — every instrument in the curated market universe, grouped by asset class (Stocks,
  Crypto, Forex, Indices, Commodities), each with its direction/rating/predicted price in one scan.
  Click any row for a live intraday candlestick chart of that instrument (Yahoo Finance intraday bars,
  auto-refreshed every 60 seconds while the tab is open) — genuinely live for crypto/forex, which trade
  around the clock; stocks and indices only move during market hours.

First prediction/backtest for a ticker trains a model (takes a few seconds to a minute); later runs
reuse the saved model until you hit Retrain.

### API

- `GET /predict?symbol=AAPL` — direction, confidence, predicted price, model metrics, trading signal
  (rating, indicator breakdown, stop-loss/take-profit, suggested position size)
- `GET /watchlist?symbols=AAPL,MSFT,GOOGL` — predicts each ticker independently (max 40; one bad
  ticker doesn't fail the batch)
- `GET /backtest?symbol=AAPL&lookback_years=5&retrain_every_days=20&confidence_threshold=0.5` —
  walk-forward backtest, returns metrics + an equity curve
- `POST /train?symbol=AAPL&lookback_years=5&horizon_days=1` — (re)trains and saves the model
- `GET /history?symbol=AAPL&days=180` — combined OHLCV history for charting
- `GET /predictions/history?symbol=AAPL` — logged predictions + resolved outcomes + calibration buckets (symbol optional)
- `GET /compare?symbols=AAPL,MSFT,SPY&days=180` — normalized (% change) price series for multiple tickers, for overlay charting
- `GET /universe` — the curated market universe (symbol, name, asset class)
- `GET /intraday?symbol=AAPL&interval=5m` — intraday OHLC candles from yfinance for the live chart
  (`interval`: `1m`/`2m`/`5m`/`15m`/`30m`/`60m`)
- `GET /health`

All of `/predict`, `/train`, and `/backtest` accept `refresh=true` to bypass the data cache.

### Daily automation (macOS)

Two scheduled `launchd` agents, both driven by the market universe (`app/universe.py`) so there's one
list to edit rather than two:

- `scripts/daily_predict.sh` runs `stockpred watchlist` (all ~20 universe instruments) weekdays at 6pm
  local time — this is what builds up the History tab over time, since predictions otherwise only
  happen when someone opens the app.
- `scripts/weekly_retrain.sh` runs `stockpred train-all --refresh` Sunday evenings — a full
  hyperparameter search + calibration retrain across the whole universe with fresh data, on its own
  schedule since it takes far longer than a same-day prediction.

```bash
cp scripts/com.atabuhmarcus.stockprediction.dailypredict.plist ~/Library/LaunchAgents/
cp scripts/com.atabuhmarcus.stockprediction.weeklyretrain.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.atabuhmarcus.stockprediction.dailypredict.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.atabuhmarcus.stockprediction.weeklyretrain.plist
```

Edit the `Hour`/`Minute`/`Weekday` values in either plist to change its schedule, then `launchctl
bootout` and re-`bootstrap` that one to apply. Output logs to `prediction_log/daily_run.log` and
`prediction_log/weekly_retrain.log` respectively. To check status, run one on demand, or remove it:

```bash
launchctl print gui/$(id -u)/com.atabuhmarcus.stockprediction.dailypredict      # status (swap in .weeklyretrain for the other job)
launchctl kickstart gui/$(id -u)/com.atabuhmarcus.stockprediction.dailypredict  # run now
launchctl bootout gui/$(id -u)/com.atabuhmarcus.stockprediction.dailypredict    # stop + remove
```

Both only run while your Mac is on and awake at the scheduled time — a missed run (laptop asleep, off,
etc.) is just skipped, not queued. If you need them to run regardless of your machine's state, that
needs a cloud runner (e.g. a scheduled GitHub Actions workflow) instead.

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
  data/                per-provider fetchers, aggregator (combine), cache, news_sentiment.py
  features/            technical indicators + labeled feature matrix (incl. market-context, sentiment)
  models/               train.py, predict.py, backtest.py, model_factory.py (tuning + calibration),
                         prediction_log.py (history + reconciliation)
  cli.py                 train / predict / watchlist / backtest / history / serve subcommands
  main.py                FastAPI app (serves the API and the web/ frontend)
web/                     browser frontend (Predict / Watchlist / Backtest / History tabs)
notebooks/               interactive exploration
models_store/            saved model bundles (*.joblib, gitignored)
data_cache/               per-ticker OHLCV + sentiment cache, refreshed daily (gitignored)
prediction_log/           prediction history log (gitignored)
```
