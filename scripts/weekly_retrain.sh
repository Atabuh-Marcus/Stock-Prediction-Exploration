#!/bin/zsh
# Retrains every model in the curated market universe (app/universe.py) with a
# fresh data pull. Invoked automatically by launchd (see
# scripts/com.atabuhmarcus.stockprediction.weeklyretrain.plist), but safe to run
# manually too: ./scripts/weekly_retrain.sh
#
# Separate from the daily prediction job on purpose: a full RandomizedSearchCV +
# calibration retrain across ~20 instruments takes far longer than a same-day
# prediction, so it runs weekly (see the plist for the schedule) rather than daily.

set -euo pipefail

PROJECT_DIR="/Users/atabuhmarcus/Stocks_Prediction"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Weekly retrain run: $(date) ==="
python -m app.cli train-all --refresh
echo ""
