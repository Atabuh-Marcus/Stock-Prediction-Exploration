const tickerInput = document.getElementById("ticker");
const predictBtn = document.getElementById("predictBtn");
const trainBtn = document.getElementById("trainBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const metaJsonEl = document.getElementById("metaJson");

let priceChart = null;
let backtestChart = null;

function setStatus(message, isError = false) {
  statusEl.hidden = !message;
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
  predictBtn.disabled = busy;
  trainBtn.disabled = busy;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return body;
}

function formatSignedPct(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

const SIGNAL_DEFINITIONS = [
  { key: "rsi_14d", label: "RSI (14d)", render: (v) => v.toFixed(1) },
  { key: "spy_relative_return_1d", label: "vs S&P 500 (1d)", render: (v) => formatSignedPct(v * 100), signed: true },
  { key: "spy_relative_return_5d", label: "vs S&P 500 (5d)", render: (v) => formatSignedPct(v * 100), signed: true },
  { key: "spy_relative_return_10d", label: "vs S&P 500 (10d)", render: (v) => formatSignedPct(v * 100), signed: true },
  { key: "market_volatility_10d", label: "Market Volatility (10d)", render: (v) => `${(v * 100).toFixed(2)}%` },
  { key: "news_sentiment", label: "News Sentiment", render: (v) => v.toFixed(3), signed: true },
  { key: "news_sentiment_5d_avg", label: "News Sentiment (5d avg)", render: (v) => v.toFixed(3), signed: true },
  { key: "news_volume_10d", label: "News Volume (10d)", render: (v) => `${Math.round(v)} articles` },
];

function renderSignals(signals) {
  const grid = document.getElementById("signalsGrid");
  grid.innerHTML = "";
  SIGNAL_DEFINITIONS.forEach((def) => {
    const value = signals ? signals[def.key] : undefined;
    if (value === undefined || value === null) return;
    const item = document.createElement("div");
    item.className = "signal-item";
    const valueClass = def.signed ? (value >= 0 ? "positive" : "negative") : "";
    item.innerHTML = `<span class="signal-label">${def.label}</span><span class="signal-value ${valueClass}">${def.render(value)}</span>`;
    grid.appendChild(item);
  });
}

const RATING_CLASSES = {
  "Strong Buy": "strong-buy",
  Buy: "buy",
  Hold: "hold",
  Sell: "sell",
  "Strong Sell": "strong-sell",
};

function renderTradingSignal(ts) {
  const badge = document.getElementById("ratingBadge");
  badge.textContent = ts.rating;
  badge.className = `rating-badge ${RATING_CLASSES[ts.rating] || ""}`;
  document.getElementById("ratingScore").textContent =
    `score ${ts.composite_score >= 0 ? "+" : ""}${ts.composite_score.toFixed(2)} · ` +
    `${ts.bullish_count} bullish / ${ts.bearish_count} bearish / ${ts.neutral_count} neutral indicators`;

  const setupGrid = document.getElementById("tradeSetupGrid");
  if (ts.stop_loss !== null && ts.take_profit !== null) {
    document.getElementById("tsEntry").textContent = `$${ts.entry_price.toFixed(2)}`;
    document.getElementById("tsStopLoss").textContent = `$${ts.stop_loss.toFixed(2)}`;
    document.getElementById("tsTakeProfit").textContent = `$${ts.take_profit.toFixed(2)}`;
    document.getElementById("tsPositionPct").textContent = `${ts.suggested_position_pct.toFixed(2)}%`;
    setupGrid.hidden = false;
  } else {
    setupGrid.hidden = true;
  }
  document.getElementById("tsSizingNote").textContent = ts.position_sizing_note;

  const list = document.getElementById("indicatorList");
  list.innerHTML = "";
  ts.indicators.forEach((ind) => {
    const row = document.createElement("div");
    row.className = "indicator-row";
    row.innerHTML =
      `<span class="indicator-marker ${ind.verdict}"></span>` +
      `<span class="indicator-name">${ind.name}</span>` +
      `<span class="indicator-detail">${ind.detail}</span>`;
    list.appendChild(row);
  });

  document.getElementById("tsNote").textContent = ts.note;
}

function renderCalibrationNote(metrics) {
  const note = document.getElementById("calibrationNote");
  const brier = metrics ? metrics.classification_brier_score : undefined;
  if (brier === undefined) {
    note.textContent = "";
    return;
  }
  let quality;
  if (brier < 0.2) quality = "reasonably well calibrated";
  else if (brier <= 0.26) quality = "near coin-flip — treat confidence loosely";
  else quality = "worse than random on its test set — treat with real caution";
  note.textContent = `Model calibration (Brier score): ${brier.toFixed(4)} — ${quality}. (0.25 = coin flip, 0 = perfect)`;
}

function renderPrediction(result) {
  resultEl.hidden = false;

  const directionEl = document.getElementById("directionValue");
  directionEl.textContent = result.direction === "rise" ? "▲ RISE" : "▼ FALL";
  directionEl.className = `direction-value ${result.direction}`;
  document.getElementById("confidenceValue").textContent =
    `${(result.direction_confidence * 100).toFixed(1)}% confidence, ${result.horizon_days}d horizon`;

  document.getElementById("lastClose").textContent = `$${result.last_close.toFixed(2)}`;

  document.getElementById("predictedPrice").textContent = `$${result.predicted_price.toFixed(2)}`;
  const changeEl = document.getElementById("predictedChange");
  const sign = result.predicted_change_pct >= 0 ? "+" : "";
  changeEl.textContent = `${sign}${result.predicted_change_pct.toFixed(2)}%`;
  changeEl.style.color = result.predicted_change_pct >= 0 ? "var(--up)" : "var(--down)";

  document.getElementById("asOf").textContent = result.as_of;

  renderTradingSignal(result.trading_signal);
  renderSignals(result.signals);
  renderCalibrationNote(result.model_metrics);

  metaJsonEl.textContent = JSON.stringify(
    { model_metrics: result.model_metrics, data_sources: result.data_sources },
    null,
    2
  );
}

function renderChart(points) {
  const ctx = document.getElementById("priceChart");
  const labels = points.map((p) => p.date);
  const closes = points.map((p) => p.close);

  if (priceChart) priceChart.destroy();
  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: closes,
          borderColor: "#4f7cff",
          backgroundColor: "rgba(79, 124, 255, 0.1)",
          fill: true,
          tension: 0.15,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b93a7", maxTicksLimit: 8 }, grid: { display: false } },
        y: { ticks: { color: "#8b93a7" }, grid: { color: "#232838" } },
      },
    },
  });
}

async function loadHistory(ticker) {
  const history = await fetchJson(`/history?symbol=${encodeURIComponent(ticker)}&days=180`);
  renderChart(history.points);
}

async function runPredict() {
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!ticker) return setStatus("Enter a ticker symbol first.", true);

  setBusy(true);
  setStatus(`Predicting ${ticker}... (first run trains a model, may take a moment)`);
  try {
    const result = await fetchJson(`/predict?symbol=${encodeURIComponent(ticker)}`);
    renderPrediction(result);
    await loadHistory(ticker);
    setStatus("");
  } catch (err) {
    resultEl.hidden = true;
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function runTrain() {
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!ticker) return setStatus("Enter a ticker symbol first.", true);

  setBusy(true);
  setStatus(`Training ${ticker}... this can take up to a minute.`);
  try {
    await fetchJson(`/train?symbol=${encodeURIComponent(ticker)}`, { method: "POST" });
    setStatus(`Model retrained for ${ticker}. Predicting...`);
    await runPredict();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

predictBtn.addEventListener("click", runPredict);
trainBtn.addEventListener("click", runTrain);
tickerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runPredict();
});

// --- Tabs ---
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = true));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).hidden = false;
  });
});

// --- Watchlist ---
const WATCHLIST_STORAGE_KEY = "stockpred.watchlist";
const watchlistTickerInput = document.getElementById("watchlistTicker");
const watchlistAddBtn = document.getElementById("watchlistAddBtn");
const watchlistRunBtn = document.getElementById("watchlistRunBtn");
const watchlistTagsEl = document.getElementById("watchlistTags");
const watchlistStatusEl = document.getElementById("watchlistStatus");
const watchlistTableCard = document.getElementById("watchlistTableCard");
const watchlistTableBody = document.getElementById("watchlistTableBody");

function loadWatchlist() {
  try {
    const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveWatchlist(tickers) {
  try {
    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(tickers));
  } catch {
    // localStorage unavailable (private mode etc) — watchlist just won't persist
  }
}

function renderWatchlistTags() {
  const tickers = loadWatchlist();
  watchlistTagsEl.innerHTML = "";
  tickers.forEach((ticker) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = ticker;
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => {
      saveWatchlist(loadWatchlist().filter((t) => t !== ticker));
      renderWatchlistTags();
    });
    tag.appendChild(removeBtn);
    watchlistTagsEl.appendChild(tag);
  });
}

function setWatchlistStatus(message, isError = false) {
  watchlistStatusEl.hidden = !message;
  watchlistStatusEl.textContent = message || "";
  watchlistStatusEl.classList.toggle("error", isError);
}

watchlistAddBtn.addEventListener("click", () => {
  const ticker = watchlistTickerInput.value.trim().toUpperCase();
  if (!ticker) return;
  const tickers = loadWatchlist();
  if (!tickers.includes(ticker)) {
    tickers.push(ticker);
    saveWatchlist(tickers);
    renderWatchlistTags();
  }
  watchlistTickerInput.value = "";
});

watchlistTickerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") watchlistAddBtn.click();
});

watchlistRunBtn.addEventListener("click", async () => {
  const tickers = loadWatchlist();
  if (tickers.length === 0) return setWatchlistStatus("Add at least one ticker first.", true);

  watchlistRunBtn.disabled = true;
  watchlistTableCard.hidden = true;
  setWatchlistStatus(`Running watchlist (${tickers.length} tickers)... first run per ticker trains a model.`);
  try {
    const entries = await fetchJson(`/watchlist?symbols=${encodeURIComponent(tickers.join(","))}`);
    watchlistTableBody.innerHTML = "";
    entries.forEach((entry) => {
      const row = document.createElement("tr");
      if (entry.error) {
        row.innerHTML = `<td>${entry.symbol}</td><td colspan="5" style="color: var(--down)">${entry.error}</td>`;
      } else {
        const p = entry.prediction;
        const arrow = p.direction === "rise" ? "▲" : "▼";
        const changeSign = p.predicted_change_pct >= 0 ? "+" : "";
        const ratingClass = RATING_CLASSES[p.trading_signal.rating] || "";
        row.innerHTML = `
          <td>${entry.symbol}</td>
          <td class="direction-value ${p.direction}" style="font-size: 0.9rem">${arrow} ${p.direction.toUpperCase()}</td>
          <td>${(p.direction_confidence * 100).toFixed(1)}%</td>
          <td><span class="rating-badge small ${ratingClass}">${p.trading_signal.rating}</span></td>
          <td>$${p.last_close.toFixed(2)}</td>
          <td>$${p.predicted_price.toFixed(2)}</td>
          <td style="color: ${p.predicted_change_pct >= 0 ? "var(--up)" : "var(--down)"}">${changeSign}${p.predicted_change_pct.toFixed(2)}%</td>
        `;
      }
      watchlistTableBody.appendChild(row);
    });
    watchlistTableCard.hidden = false;

    setWatchlistStatus("Loading comparison chart...");
    try {
      const compareData = await fetchJson(`/compare?symbols=${encodeURIComponent(tickers.join(","))}&days=180`);
      renderCompareChart(compareData.series);
    } catch {
      // comparison chart is a nice-to-have on top of the table — don't fail the whole run over it
      document.getElementById("compareChartCard").hidden = true;
    }
    setWatchlistStatus("");
  } catch (err) {
    setWatchlistStatus(err.message, true);
  } finally {
    watchlistRunBtn.disabled = false;
  }
});

let compareChart = null;
const COMPARE_COLORS = ["#4f7cff", "#22c55e", "#ef4444", "#eab308", "#a855f7", "#06b6d4", "#f97316", "#ec4899"];

function renderCompareChart(series) {
  const card = document.getElementById("compareChartCard");
  const validSeries = series.filter((s) => !s.error && s.points.length > 0);
  if (validSeries.length === 0) {
    card.hidden = true;
    return;
  }

  const labels = validSeries[0].points.map((p) => p.date);
  const ctx = document.getElementById("compareChart");
  if (compareChart) compareChart.destroy();
  compareChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: validSeries.map((s, i) => ({
        label: s.symbol,
        data: s.points.map((p) => p.pct_change),
        borderColor: COMPARE_COLORS[i % COMPARE_COLORS.length],
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.1,
      })),
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true, labels: { color: "#e6e9f0" } } },
      scales: {
        x: { ticks: { color: "#8b93a7", maxTicksLimit: 8 }, grid: { display: false } },
        y: { ticks: { color: "#8b93a7", callback: (v) => `${v}%` }, grid: { color: "#232838" } },
      },
    },
  });
  card.hidden = false;
}

renderWatchlistTags();

// --- Backtest ---
const backtestTickerInput = document.getElementById("backtestTicker");
const backtestRunBtn = document.getElementById("backtestRunBtn");
const backtestStatusEl = document.getElementById("backtestStatus");
const backtestResultEl = document.getElementById("backtestResult");

function setBacktestStatus(message, isError = false) {
  backtestStatusEl.hidden = !message;
  backtestStatusEl.textContent = message || "";
  backtestStatusEl.classList.toggle("error", isError);
}

function renderBacktestChart(equityCurve) {
  const ctx = document.getElementById("backtestChart");
  const labels = equityCurve.map((p) => p.date);

  if (backtestChart) backtestChart.destroy();
  backtestChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Strategy",
          data: equityCurve.map((p) => p.strategy_cum_return_pct),
          borderColor: "#4f7cff",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.1,
        },
        {
          label: "Buy & Hold",
          data: equityCurve.map((p) => p.buy_hold_cum_return_pct),
          borderColor: "#8b93a7",
          borderDash: [4, 4],
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true, labels: { color: "#e6e9f0" } } },
      scales: {
        x: { ticks: { color: "#8b93a7", maxTicksLimit: 8 }, grid: { display: false } },
        y: {
          ticks: { color: "#8b93a7", callback: (v) => `${v}%` },
          grid: { color: "#232838" },
        },
      },
    },
  });
}

backtestRunBtn.addEventListener("click", async () => {
  const ticker = backtestTickerInput.value.trim().toUpperCase();
  if (!ticker) return setBacktestStatus("Enter a ticker symbol first.", true);

  const lookbackYears = document.getElementById("btLookback").value;
  const retrainEvery = document.getElementById("btRetrainEvery").value;
  const confidence = document.getElementById("btConfidence").value;

  backtestRunBtn.disabled = true;
  backtestResultEl.hidden = true;
  setBacktestStatus(`Backtesting ${ticker}... this retrains many times and can take a minute.`);
  try {
    const params = new URLSearchParams({
      symbol: ticker,
      lookback_years: lookbackYears,
      retrain_every_days: retrainEvery,
      confidence_threshold: confidence,
    });
    const result = await fetchJson(`/backtest?${params.toString()}`);

    document.getElementById("btHitRate").textContent = `${(result.hit_rate * 100).toFixed(1)}%`;
    document.getElementById("btStrategyReturn").textContent = `${result.strategy_total_return_pct >= 0 ? "+" : ""}${result.strategy_total_return_pct.toFixed(2)}%`;
    document.getElementById("btBuyHoldReturn").textContent = `${result.buy_hold_total_return_pct >= 0 ? "+" : ""}${result.buy_hold_total_return_pct.toFixed(2)}%`;
    document.getElementById("btSharpe").textContent = result.strategy_sharpe.toFixed(2);
    document.getElementById("btMaxDrawdown").textContent = `${result.strategy_max_drawdown_pct.toFixed(2)}%`;
    document.getElementById("btDaysInMarket").textContent = `${result.days_in_market_pct.toFixed(1)}%`;
    document.getElementById("backtestNote").textContent = `${result.note} (${result.start_date} to ${result.end_date}, ${result.evaluated_days} days, ${result.retrains} retrains)`;

    renderBacktestChart(result.equity_curve);
    backtestResultEl.hidden = false;
    setBacktestStatus("");
  } catch (err) {
    setBacktestStatus(err.message, true);
  } finally {
    backtestRunBtn.disabled = false;
  }
});

// --- History ---
const historyTickerInput = document.getElementById("historyTicker");
const historyLoadBtn = document.getElementById("historyLoadBtn");
const historyStatusEl = document.getElementById("historyStatus");
const historyResultEl = document.getElementById("historyResult");

function setHistoryStatus(message, isError = false) {
  historyStatusEl.hidden = !message;
  historyStatusEl.textContent = message || "";
  historyStatusEl.classList.toggle("error", isError);
}

historyLoadBtn.addEventListener("click", async () => {
  const ticker = historyTickerInput.value.trim().toUpperCase();

  historyLoadBtn.disabled = true;
  historyResultEl.hidden = true;
  setHistoryStatus("Loading prediction history...");
  try {
    const query = ticker ? `?symbol=${encodeURIComponent(ticker)}` : "";
    const result = await fetchJson(`/predictions/history${query}`);

    document.getElementById("histTotal").textContent = result.total_predictions;
    document.getElementById("histResolved").textContent = result.resolved_predictions;
    document.getElementById("histAccuracy").textContent =
      result.accuracy === null ? "—" : `${(result.accuracy * 100).toFixed(1)}%`;

    const calTableCard = document.getElementById("calibrationTableCard");
    const calBody = document.getElementById("calibrationTableBody");
    calBody.innerHTML = "";
    if (result.calibration_buckets.length > 0) {
      result.calibration_buckets.forEach((b) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${b.range}</td><td>${b.count}</td><td>${(b.actual_accuracy * 100).toFixed(1)}%</td>`;
        calBody.appendChild(row);
      });
      calTableCard.hidden = false;
    } else {
      calTableCard.hidden = true;
    }

    const rowsBody = document.getElementById("historyTableBody");
    rowsBody.innerHTML = "";
    if (result.rows.length === 0) {
      rowsBody.innerHTML =
        '<tr><td colspan="8" class="sub">No predictions logged yet — use Predict or Watchlist to start building history.</td></tr>';
    } else {
      result.rows.forEach((row) => {
        const tr = document.createElement("tr");
        const statusText = row.resolved ? "resolved" : "pending";
        let resultText = "—";
        let resultColor = "var(--muted)";
        if (row.correct === true) {
          resultText = "✓ correct";
          resultColor = "var(--up)";
        } else if (row.correct === false) {
          resultText = "✗ wrong";
          resultColor = "var(--down)";
        }
        tr.innerHTML = `
          <td>${row.ticker}</td>
          <td>${row.as_of_date}</td>
          <td>${row.predicted_direction === "rise" ? "▲ rise" : "▼ fall"}</td>
          <td>${(row.confidence * 100).toFixed(1)}%</td>
          <td>${statusText}</td>
          <td>$${row.predicted_price.toFixed(2)}</td>
          <td>${row.actual_close !== null ? "$" + row.actual_close.toFixed(2) : "—"}</td>
          <td style="color: ${resultColor}">${resultText}</td>
        `;
        rowsBody.appendChild(tr);
      });
    }

    historyResultEl.hidden = false;
    setHistoryStatus("");
  } catch (err) {
    setHistoryStatus(err.message, true);
  } finally {
    historyLoadBtn.disabled = false;
  }
});
