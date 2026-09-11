#!/bin/zsh
# Runs the daily watchlist prediction. Invoked automatically by launchd
# (see scripts/com.atabuhmarcus.stockprediction.dailypredict.plist), but safe
# to run manually too: ./scripts/daily_predict.sh
#
# Edit the ticker list below to track different symbols.

set -euo pipefail

PROJECT_DIR="/Users/atabuhmarcus/Stocks_Prediction"
TICKERS=(AAPL MSFT GOOGL AMZN)

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Daily prediction run: $(date) ==="
python -m app.cli watchlist "${TICKERS[@]}"
echo ""
