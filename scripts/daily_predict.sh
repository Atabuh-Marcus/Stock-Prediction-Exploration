#!/bin/zsh
# Runs the daily watchlist prediction across the full curated market universe
# (app/universe.py). Invoked automatically by launchd (see
# scripts/com.atabuhmarcus.stockprediction.dailypredict.plist), but safe to run
# manually too: ./scripts/daily_predict.sh
#
# To track a different/smaller set instead, pass explicit tickers to
# `python -m app.cli watchlist` below — with none given it defaults to the universe.

set -euo pipefail

PROJECT_DIR="/Users/atabuhmarcus/Stocks_Prediction"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Daily prediction run: $(date) ==="
python -m app.cli watchlist
echo ""
