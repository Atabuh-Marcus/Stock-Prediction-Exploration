from __future__ import annotations

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

PARAM_DISTRIBUTIONS = {
    "max_iter": randint(50, 300),
    "max_depth": [None, 3, 5, 7, 10],
    "learning_rate": uniform(0.01, 0.29),
    "l2_regularization": uniform(0.0, 1.0),
    "min_samples_leaf": randint(10, 60),
}


def _cv_splits(n_rows: int, target: int) -> int:
    """Caps the number of CV folds so each fold still has a reasonable number of
    rows — a short history (small-cap tickers, short lookback) shouldn't get more
    folds than it can support."""
    return max(2, min(target, n_rows // 50))


def tune_classifier(X: pd.DataFrame, y: pd.Series, n_splits: int = 4, n_iter: int = 15, random_state: int = 42) -> dict:
    """Randomized search over HistGradientBoostingClassifier hyperparameters using
    time-series cross-validation — folds respect chronological order (no
    shuffling), so no future data ever leaks into a training fold."""
    cv = TimeSeriesSplit(n_splits=_cv_splits(len(X), n_splits))
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=random_state),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_params_


def tune_regressor(X: pd.DataFrame, y: pd.Series, n_splits: int = 4, n_iter: int = 15, random_state: int = 42) -> dict:
    cv = TimeSeriesSplit(n_splits=_cv_splits(len(X), n_splits))
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_params_


def fit_calibrated_classifier(
    X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int = 3, random_state: int = 42
) -> CalibratedClassifierCV:
    """Fits a HistGradientBoostingClassifier with the given (tuned) hyperparameters,
    wrapped in CalibratedClassifierCV so predict_proba is actually trustworthy — an
    uncalibrated model can say "70% confident" while being right only ~55% of the
    time. Uses sigmoid (Platt) scaling rather than isotonic, which needs more data
    than a single-ticker daily-bar dataset typically has to avoid overfitting.
    """
    cv = TimeSeriesSplit(n_splits=_cv_splits(len(X), n_splits))
    base = HistGradientBoostingClassifier(random_state=random_state, **params)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    calibrated.fit(X, y)
    return calibrated


def fit_regressor(X: pd.DataFrame, y: pd.Series, params: dict, random_state: int = 42) -> HistGradientBoostingRegressor:
    regressor = HistGradientBoostingRegressor(random_state=random_state, **params)
    regressor.fit(X, y)
    return regressor
