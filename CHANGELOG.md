# Changelog

Build history for this project — what was built, what broke, and how it was fixed. `README.md` covers
how to use the finished thing; this covers how it got here.

## 1. Initial build

Scaffolded around an existing `app/main.py` starter (a single-file FastAPI + yfinance + logistic
regression prototype) rather than replacing it — extended it into the full structure below.

- **Data layer** (`app/data/`): three providers behind a common `DataSource` interface —
  `yfinance_source.py` (free, no key), `alpha_vantage_source.py`, `polygon_source.py` (both optional,
  activate automatically once an API key is set). `aggregator.py` fetches from every configured
  source and combines them by taking the per-day median across whatever responded, so one source
  glitching doesn't skew the result.
- **Features** (`app/features/indicators.py`, `build_features.py`): SMA/EMA gaps, RSI, MACD,
  Bollinger bandwidth, rolling volatility, volume z-score.
- **Models** (`app/models/train.py`, `predict.py`): a `HistGradientBoostingClassifier` (direction)
  and `HistGradientBoostingRegressor` (price) per ticker, time-based train/test split, saved to
  `models_store/<TICKER>.joblib`.
- **Interfaces**: CLI (`app/cli.py`: `train` / `predict` / `serve`), FastAPI backend + browser
  frontend (`app/main.py`, `web/`), and a Jupyter notebook (`notebooks/exploration.ipynb`) — all
  built on the same data/features/models modules so results match across all three.

## 2. Bugs found once real data and real API keys were in play

Everything below was caught by actually running the thing against live providers, not by inspection.

- **`.env.example` had real API keys pasted into it** instead of `.env`. Since `.env.example` isn't
  gitignored (it's meant to be a safe template), this would have leaked the keys the moment the repo
  was committed. Restored the template to empty placeholders; real keys live only in `.env`.
- **Alpha Vantage's free tier no longer allows `outputsize=full`** (now a paid-plan feature) — every
  request was failing with a rate-limit-shaped error that was actually a plan restriction. Switched
  to `outputsize=compact` (last ~100 trading days, what the free tier actually allows).
- **`.env` loading was working-directory-dependent.** Fine from the project root, silently found
  nothing when Jupyter's kernel started from `notebooks/`. Fixed by resolving the path from
  `app/config.py`'s own location instead of relying on `load_dotenv()`'s default search.
- **The regression model was fundamentally miscalibrated.** It predicted the *absolute* next-day
  price using a feature set that's almost entirely scale-invariant (returns, ratios, RSI) — no
  feature told it what today's actual price was, so it had no way to anchor its output. It once
  predicted a 24% overnight move on AAPL (test-set MAE ~$70). Fixed by training the regressor on
  next-period **return** instead and reconstructing `price = last_close * (1 + predicted_return)` at
  inference time (`app/features/build_features.py`, `app/models/train.py::_evaluate`,
  `app/models/predict.py`). Also normalized MACD by price for the same reason (it's an absolute
  dollar value by construction, which drifts as a stock's price level changes over a multi-year
  window). MAE dropped from ~$70 to ~$4.50 on the same AAPL data after the fix.
- **yfinance intermittently returns an empty DataFrame with no exception** (bot-detection /
  rate-limiting, not a hard failure). Hardened `yfinance_source.py` to try two different yfinance
  call paths (`Ticker().history()` then `download()`), each retried once with a short backoff, with a
  clearer error message when both fail.

## 3. Data caching, backtesting, and watchlist

- **Data caching** (`app/data/cache.py`): combined OHLCV is cached to `data_cache/<TICKER>.csv`.
  A cache written earlier today is treated as fresh (daily bars don't change intraday), so repeated
  predict/train/backtest calls stop re-hitting rate-limited free-tier APIs. `refresh=True` /
  `--refresh` bypasses it.
  - **Bug found while wiring this up**: the initial cache-coverage check required the cache to reach
    back to the *exact* requested start date. Polygon's free tier caps history at ~2 years, and the
    requested start date is deliberately padded by 30 days (for weekends/holidays) — so the cache
    could never satisfy the strict check and silently refetched from the network on every single
    call, defeating the entire point of caching. Fixed with a tolerance window (`cache.py::covers`).
- **Walk-forward backtesting** (`app/models/backtest.py`): retrains periodically using only data
  available up to that point (no lookahead), then simulates a simple long/cash strategy — long when
  predicted direction is "rise" with confidence over a threshold, flat otherwise — against a buy &
  hold benchmark. Reports total return, Sharpe, max drawdown, days in market, and out-of-sample
  directional hit rate. Exposed via `app/cli.py::backtest`, `GET /backtest`, and a Backtest tab in
  the web app (equity-curve chart: strategy vs. buy & hold).
- **Multi-ticker watchlist** (`app/models/predict.py::predict_watchlist`): predicts a list of tickers
  independently — one bad/rate-limited ticker doesn't fail the batch. Exposed via
  `app/cli.py::watchlist`, `GET /watchlist`, and a Watchlist tab (tickers persist in the browser via
  `localStorage`).

## 4. Testing performed

- Synthetic-data smoke tests (no network) for the feature engineering, train/predict pipeline, the
  backtest core loop, and the cache layer.
- Live tests against the user's real Alpha Vantage and Polygon API keys (train, predict, watchlist,
  backtest all confirmed working against real market data).
- A full browser walkthrough via Playwright (headless Chromium) driving the actual served pages —
  Predict, Watchlist, and Backtest tabs, screenshotted at each step.
  - **Bug caught only by this**: the price chart had been broken since the very first build. The
    Chart.js CDN URL used a version that doesn't exist (`4.4.4`) and, once corrected to the real
    latest version, the default `chart.min.js` build turned out to be ES-module-only (`import`
    statements), which throws when loaded via a plain `<script src>` tag. Needed the explicit UMD
    build: `.../Chart.js/4.5.1/chart.umd.min.js`. Earlier verification had only checked HTTP status
    codes and byte counts of the static files — never actually executed the JS — so this sat broken
    through the whole initial build and first round of fixes.

## Known limitations

- Next-day directional accuracy is close to random (~50–52% in testing) — expected for a model using
  only technical indicators on daily bars; treat all outputs as informational, not a trading signal.
- yfinance availability depends on network/IP (Yahoo does bot-detection and rate-limiting); Alpha
  Vantage's free tier is capped at ~100 days of history and 25 requests/day; Polygon's free tier caps
  history at roughly 2 years.
- The backtest strategy is a simplified long/cash simulation — no shorting, fees, slippage, or
  position sizing.
