from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODELS_DIR = BASE_DIR / "models_store"
DATA_CACHE_DIR = BASE_DIR / "data_cache"
PREDICTION_LOG_DIR = BASE_DIR / "prediction_log"
MODELS_DIR.mkdir(exist_ok=True)
DATA_CACHE_DIR.mkdir(exist_ok=True)
PREDICTION_LOG_DIR.mkdir(exist_ok=True)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()

# Horizon (in trading days) the models predict ahead.
PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", "1"))

# How much history to pull when training from scratch.
DEFAULT_LOOKBACK_YEARS = int(os.getenv("DEFAULT_LOOKBACK_YEARS", "5"))
