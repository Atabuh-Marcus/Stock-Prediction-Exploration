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
        row.innerHTML = `
          <td>${entry.symbol}</td>
          <td class="direction-value ${p.direction}" style="font-size: 0.9rem">${arrow} ${p.direction.toUpperCase()}</td>
          <td>${(p.direction_confidence * 100).toFixed(1)}%</td>
          <td>$${p.last_close.toFixed(2)}</td>
          <td>$${p.predicted_price.toFixed(2)}</td>
          <td style="color: ${p.predicted_change_pct >= 0 ? "var(--up)" : "var(--down)"}">${changeSign}${p.predicted_change_pct.toFixed(2)}%</td>
        `;
      }
      watchlistTableBody.appendChild(row);
    });
    watchlistTableCard.hidden = false;
    setWatchlistStatus("");
  } catch (err) {
    setWatchlistStatus(err.message, true);
  } finally {
    watchlistRunBtn.disabled = false;
  }
});

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
