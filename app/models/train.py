from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score

from app.config import DEFAULT_LOOKBACK_YEARS, MODELS_DIR, PREDICTION_HORIZON_DAYS
from app.data.aggregator import DataAggregator, default_lookback_start
from app.features.build_features import FEATURE_COLUMNS, build_training_dataset

TEST_FRACTION = 0.2


@dataclass
class ModelBundle:
    ticker: str
    horizon_days: int
    feature_columns: list[str]
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    source_status: dict = field(default_factory=dict)
    last_known_close: float = 0.0
    last_known_date: str = ""


def _time_based_split(n_rows: int, test_fraction: float) -> int:
    split_at = int(n_rows * (1 - test_fraction))
    return max(split_at, 30)  # keep a usable training set even on short histories


def train_ticker(
    ticker: str,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
    refresh: bool = False,
) -> ModelBundle:
    ticker = ticker.upper().strip()
    aggregator = DataAggregator()
    ohlcv, source_status = aggregator.fetch(
        ticker, start=default_lookback_start(lookback_years), use_cache=not refresh
    )

    features, class_target, reg_target = build_training_dataset(ohlcv, horizon_days)
    if len(features) < 60:
        raise ValueError(
            f"Only {len(features)} usable rows of history for '{ticker}' — need at least 60. "
            "Try a longer lookback period."
        )

    split = _time_based_split(len(features), TEST_FRACTION)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_class_train, y_class_test = class_target.iloc[:split], class_target.iloc[split:]
    y_reg_train, y_reg_test = reg_target.iloc[:split], reg_target.iloc[split:]

    classifier = HistGradientBoostingClassifier(random_state=42)
    classifier.fit(X_train, y_class_train)

    regressor = HistGradientBoostingRegressor(random_state=42)
    regressor.fit(X_train, y_reg_train)

    current_close_test = ohlcv["close"].loc[X_test.index]
    metrics = _evaluate(classifier, regressor, X_test, y_class_test, y_reg_test, current_close_test)

    bundle = ModelBundle(
        ticker=ticker,
        horizon_days=horizon_days,
        feature_columns=FEATURE_COLUMNS,
        classifier=classifier,
        regressor=regressor,
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(),
        source_status=source_status,
        last_known_close=float(ohlcv["close"].iloc[-1]),
        last_known_date=str(ohlcv.index[-1].date()),
    )
    _save(bundle)
    return bundle


def _evaluate(
    classifier: HistGradientBoostingClassifier,
    regressor: HistGradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_class_test: pd.Series,
    y_reg_test: pd.Series,
    current_close_test: pd.Series,
) -> dict:
    if X_test.empty:
        return {"warning": "No held-out rows available; metrics skipped (history too short)."}

    class_pred = classifier.predict(X_test)
    class_proba = classifier.predict_proba(X_test)[:, 1]
    reg_pred_return = regressor.predict(X_test)  # regressor predicts return, not price

    current_close = current_close_test.to_numpy()
    predicted_price = current_close * (1 + reg_pred_return)
    actual_price = current_close * (1 + y_reg_test.to_numpy())

    reg_direction_pred = reg_pred_return > 0
    reg_direction_actual = y_class_test.to_numpy().astype(bool)

    metrics = {
        "test_rows": int(len(X_test)),
        "classification_accuracy": round(float(accuracy_score(y_class_test, class_pred)), 4),
        "classification_f1": round(float(f1_score(y_class_test, class_pred, zero_division=0)), 4),
        "regression_mae": round(float(mean_absolute_error(actual_price, predicted_price)), 4),
        "regression_directional_accuracy": round(float(np.mean(reg_direction_pred == reg_direction_actual)), 4),
    }
    if y_class_test.nunique() > 1:
        metrics["classification_roc_auc"] = round(float(roc_auc_score(y_class_test, class_proba)), 4)
    return metrics


def _model_path(ticker: str) -> str:
    return str(MODELS_DIR / f"{ticker.upper()}.joblib")


def _save(bundle: ModelBundle) -> None:
    joblib.dump(bundle, _model_path(bundle.ticker))


def load_bundle(ticker: str) -> ModelBundle | None:
    path = MODELS_DIR / f"{ticker.upper()}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
