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

## 5. Model quality: market context, news sentiment, tuning, calibration

- **Market-context features** (`app/data/aggregator.py::fetch_benchmark`,
  `app/features/build_features.py`): fetches SPY alongside the target ticker (through the same
  aggregator/cache, so no new infrastructure) and adds the stock's return relative to SPY at
  1/5/10 days plus SPY's own rolling volatility as a market-wide risk proxy. Neutral/zero when the
  benchmark fetch fails, so this never breaks the pipeline.
- **News sentiment** (`app/data/news_sentiment.py`): daily-aggregated sentiment from Alpha Vantage's
  `NEWS_SENTIMENT` endpoint (one call per ticker per day, cached like OHLCV data), forward-filled
  between news events and smoothed with a 5-day average. Neutral/zero when no Alpha Vantage key is
  configured.
- **Hyperparameter tuning + confidence calibration** (`app/models/model_factory.py`): replaced
  default-parameter model fits with `RandomizedSearchCV` over tree depth / learning rate /
  regularization / leaf size, using `TimeSeriesSplit` cross-validation so no fold ever trains on
  data from after its validation period. The classifier is further wrapped in
  `CalibratedClassifierCV` (sigmoid/Platt scaling) so its reported confidence is checked against
  actual outcomes rather than being a raw, potentially overconfident score — tracked via a new
  `classification_brier_score` metric (0.25 = coin flip; a live AAPL run came back at 0.2497,
  honestly reflecting that next-day direction from these features is close to unpredictable, which
  is the expected and correct result, not a bug).
- **Backtest consistency**: `run_backtest` uses the same tuned + calibrated model-building path as
  `train_ticker` (via the shared `model_factory` module), so the backtest actually reflects what the
  live predict/train path does. Full hyperparameter search only runs once, on the initial
  walk-forward window (no lookahead — it's the same data the first retrain checkpoint trains on
  anyway); re-running a full search at every one of the many walk-forward retrain checkpoints would
  be far too slow, so those reuse the once-tuned settings while still calibrating fresh each time.
- Feature count went from 17 to 24; `FEATURE_COLUMNS` in `build_features.py` is the source of truth.
- **Bug found in the same pass**: `news_sentiment.py` originally requested Alpha Vantage's
  `NEWS_SENTIMENT` with `sort=EARLIEST`. For a heavily-covered ticker like AAPL (25-48 articles/day),
  the endpoint's 1000-article cap was exhausted by articles from the *start* of the lookback window
  and never reached recent dates — a live check came back with sentiment data ending 5 months ago.
  Switched to `sort=LATEST` so the article budget is spent on the most recent (most relevant for
  prediction) news instead.

## 6. Web UI: signals panel, prediction history, performance comparison

- **Signals panel** (Predict tab): the market-context, sentiment, and calibration numbers added in
  section 5 were only visible in a collapsed JSON blob. Added a readable panel (RSI, relative-to-SPY
  returns, market volatility, news sentiment, news volume) plus a plain-language calibration note
  derived from the Brier score, with the raw JSON still available underneath for anyone who wants it.
- **Prediction history tracking** (`app/models/prediction_log.py`, new History tab): every prediction
  is now logged (deduped per ticker/day) and, once its target date has passed, reconciled against the
  actual close price the next time history is viewed. Surfaces total/resolved counts, running
  accuracy, a confidence-bucketed calibration table (does "70% confident" actually land ~70% of the
  time in practice, not just in the backtest), and the full row-level log. New endpoint:
  `GET /predictions/history`; CLI: `stockpred history [TICKER]`.
- **Multi-ticker performance comparison** (Watchlist tab): running a watchlist now also renders a
  normalized (% change from period start) overlay chart across those same tickers — folded into the
  existing Watchlist tab rather than a new one, since it's naturally the same ticker set. New
  endpoint: `GET /compare?symbols=...&days=...`.
- Verified with a full Playwright browser walkthrough (not just endpoint checks) after the Chart.js
  CDN lesson from the original build — zero console errors, all three additions confirmed rendering
  with real data via screenshots.

## 7. Daily automation

The History tab only accumulates data when a prediction actually runs, which otherwise only happens
when someone opens the app — so it added a scheduled job instead of staying manual-only:

- `scripts/daily_predict.sh`: activates the venv and runs `stockpred watchlist` for a fixed ticker
  list (AAPL, MSFT, GOOGL, AMZN — edit the `TICKERS` line to change it).
- `scripts/com.atabuhmarcus.stockprediction.dailypredict.plist`: a macOS `launchd` agent running that
  script weekdays at 6pm local time, installed to `~/Library/LaunchAgents/`. Chosen over a cloud
  scheduler (e.g. GitHub Actions) for this pass since it needs no API keys leaving the machine and no
  changes to what's gitignored — the tradeoff is it only runs while the Mac is on and awake at the
  scheduled time; a missed run is skipped, not queued.
- Verified through the actual production path, not just direct script execution: installed via
  `launchctl bootstrap`, confirmed the calendar-interval schedule registered correctly for all 5
  weekdays, then fired it on demand via `launchctl kickstart` and confirmed the output and exit code
  (0) came back clean, with the prediction-log dedup logic correctly preventing duplicate rows across
  the multiple test runs done the same day.

## 8. Trading signals: rating, indicator breakdown, risk levels, position sizing

Direction + confidence + a price target is informative but not actionable — nothing said what to
actually *do* with a prediction. Added `app/models/trading_signals.py`, wired into `predict_ticker` so
every `/predict` and `/watchlist` result now carries a `trading_signal` block:

- **Composite rating** (Strong Buy / Buy / Hold / Sell / Strong Sell): a weighted blend of the ML
  confidence (60%) and agreement across six technical indicators (RSI, MACD, trend vs. 20d SMA,
  Bollinger Bands, relative strength vs. SPY, 5-day news sentiment average) at 40%. The ML side is
  weighted higher deliberately — it's the only piece actually validated via backtesting and Brier-score
  calibration; the indicator vote is a secondary confirmation/dissent check, not an equal-weight second
  model.
- **Indicator breakdown**: each of the six indicators reports bullish/bearish/neutral with a plain-
  language reason (e.g. "58.9 — neutral range"), so the rating is traceable instead of a black box.
- **Risk levels**: added `atr()` (Wilder's smoothing) to `app/features/indicators.py`. A Buy/Sell rating
  gets an ATR-based stop-loss (1.5× ATR) and take-profit (1.5:1 reward:risk); a Hold gets neither — there's
  no qualifying trade setup to size. A Sell/Strong Sell produces a short setup, explicitly flagged as an
  extrapolation from the same signals rather than something the (long-only) backtest has validated.
- **Position sizing**: a suggested position size (% of portfolio), derived from a standard 1%-of-equity
  risk-per-trade convention divided by the stop distance, scaled down as confidence approaches a coin
  flip (50%) and capped at 15% of a single position — account-size-independent since the app has no
  notion of the user's actual capital.
- Web app: new Trading Signal card on the Predict tab (rating badge, entry/stop/target/size cards,
  indicator list) and a Rating column on the Watchlist table. CLI: `predict` and `watchlist` output
  extended to match.
- Verified against real AAPL data through the full live path (`/predict`, not just the underlying
  function) and with a Playwright browser pass on both the Predict and Watchlist tabs — zero console
  errors, rating badge and indicator rows confirmed rendering with live values.

## 9. Multi-asset support: crypto, forex, indices, commodities

Every ticker field previously accepted only plain equity-style symbols. Widened to accept any
yfinance-style symbol, since the data/feature/model pipeline already had no actual dependency on the
instrument being a stock:

- Loosened the ticker validation regex (`app/main.py`) to allow `=` and `^` alongside the existing
  letters/digits/`.`/`-`, covering crypto (`BTC-USD`), forex (`EURUSD=X`), indices (`^GSPC`), and
  commodity futures (`GC=F`) symbol formats, and bumped the max length from 10 to 12 to fit them.
- **Bug found while testing this (via synthetic data, since forex data isn't reachable from this
  sandbox — see note below)**: spot forex pairs report zero volume from Yahoo (no central exchange
  tape for OTC forex), and a constant-zero volume series turns `volume_change` (`pct_change`) and
  `volume_zscore_20d` into NaN for *every* row via 0/0 division — which would silently wipe out the
  entire feature frame through the `dropna()` in `build_training_dataset`/`build_latest_feature_row`,
  making prediction impossible for any zero-volume instrument. Fixed in
  `app/features/build_features.py::_price_features` by neutralizing those two columns to 0.0 in that
  case instead of leaving them NaN — other columns' own warmup NaNs (e.g. the 50-day SMA gap) still
  trim the start of the series normally, so this doesn't mask genuine missing history.
- Web app: ticker inputs' placeholder/format hint updated with example symbols across asset classes.
- **Testing note**: this sandbox's network blocks Yahoo Finance with a TLS interception error (a
  known sandbox-specific restriction seen earlier in this project's development, not a code issue —
  see section 2) for any symbol not already cached, so BTC-USD/EURUSD=X/^GSPC/GC=F could not be
  fetched live from here. Verified instead with synthetic OHLCV (including a zero-volume series
  simulating real forex data) through the actual feature-building functions; AAPL (already cached)
  continued to predict correctly through the live server after the change. Live confirmation of a
  real crypto/forex/index fetch is still worth doing on your own machine.

## 10. Market universe, Markets dashboard, live intraday candles, regular retraining

Asked to cover more tradeable instruments in one place, with a live chart and regular retraining
across all of it. Four pieces:

- **`app/universe.py`** (new): a curated ~22-instrument list across every supported asset class
  (10 mega-cap stocks, 3 crypto, 3 forex pairs, 3 indices, 3 commodities) — not a literal "every symbol
  that exists," since no provider here has a discovery endpoint, and an unbounded list would blow
  through Alpha Vantage's 25-requests/day free quota and make retraining impractically slow. Now the
  single source of truth for the Markets tab, `stockpred watchlist` with no arguments, and the new
  `stockpred train-all`.
- **Live intraday candlestick chart** (`app/data/intraday.py`, `GET /intraday`, new Markets tab): a
  separate yfinance-only intraday fetch path (1m–60m bars) — deliberately not folded into the daily
  `DataAggregator`, since its day-level freshness cache and multi-source median combination don't apply
  to a chart that's supposed to update every poll. No free provider here offers real tick-level
  streaming, so "live" means the chart polls this endpoint every 60 seconds while the tab is open —
  genuinely live-ish for crypto/forex (trade ~24/7), market-hours-only for stocks/indices.
  - Rendered with `chartjs-chart-financial` + `chartjs-adapter-date-fns` (candlestick chart type isn't
    built into Chart.js). Neither is on cdnjs (checked directly — chartjs-chart-financial isn't listed
    there at all), so these load from jsDelivr instead; unlike the Artifact tool's publishing sandbox,
    this is a plain FastAPI-served page with no CDN allowlist. Verified both are genuine UMD bundles
    (not ESM-only) and version-compatible with the already-pinned Chart.js 4.5.1 by inspecting each
    package's `package.json` peer dependencies and the first bytes of the actual served file before
    wiring them in — the kind of check that would have caught the original Chart.js version/build
    mistake (section 3) before it shipped, rather than after.
- **Markets tab**: scans the whole universe via the existing `/watchlist` (cap raised 20 → 40 to fit
  it), grouped into per-asset-class tables; clicking a row selects that instrument's live chart.
- **Regular retraining across the universe**: `train_all()` (`app/models/train.py`) and
  `stockpred train-all` retrain every universe instrument with the same per-ticker error isolation as
  `predict_watchlist`. New `scripts/weekly_retrain.sh` + `com.atabuhmarcus.stockprediction.
  weeklyretrain.plist` run it Sunday evenings — on its own schedule and separate from the daily
  prediction job, since a full hyperparameter search + calibration across ~20 instruments takes far
  longer than a same-day prediction. `scripts/daily_predict.sh` now runs the full universe too (was a
  hardcoded 4-stock list), so the History tab's calibration data builds up across every asset class,
  not just the original stocks.
- **Testing note**: this sandbox's network blocks live Yahoo Finance calls for anything not already
  cached (see section 9), so a full live Markets scan and a genuine live intraday fetch couldn't be
  run from here. Verified instead by mocking `/universe`, `/watchlist`, and `/intraday` in a real
  browser (Playwright) with realistic response shapes — confirmed the candlestick chart renders
  correctly (including axis scaling for both an AAPL-range and a BTC-USD-range dataset), asset-class
  grouping, per-row error isolation, row-click symbol switching, and interval switching, all with zero
  console errors — plus the CLI (`watchlist`, `predict`, `train-all --help`) against already-cached
  AAPL data. A real live scan and intraday feed are still worth confirming on your own machine.

## Known limitations

- Next-day directional accuracy is close to random (~50–52% in testing) — expected for a model using
  only technical indicators on daily bars; treat all outputs as informational, not a trading signal.
- yfinance availability depends on network/IP (Yahoo does bot-detection and rate-limiting); Alpha
  Vantage's free tier is capped at ~100 days of history and 25 requests/day (now shared between price
  and news-sentiment calls — training/predicting a ticker can use 2 AV calls in one run, SPY's price
  fetch a 3rd, shared across all tickers and cached daily); Polygon's free tier caps history at
  roughly 2 years.
- The backtest strategy is a simplified long/cash simulation — no shorting, fees, slippage, or
  position sizing.
- The trading-signal rating, stop-loss/take-profit, and position sizing are a deterministic formula
  layered on top of the model's output (indicator agreement + ATR + a fixed risk-per-trade convention) —
  they are not separately backtested or calibrated the way the direction/price predictions are, and the
  short-side setup on a Sell rating is a symmetric extrapolation, not something the long-only backtest
  has validated.
- The Markets tab's "live" chart is 60-second polling of intraday bars, not real-time tick streaming —
  no free data provider here offers that. The market universe is a fixed curated list (`app/universe.py`),
  not every instrument that exists; predictions and the trading signal still carry the same ~50–52%
  directional accuracy regardless of asset class.
