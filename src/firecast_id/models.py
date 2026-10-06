"""Forecast models behind one interface.

Every model has ``fit(train, valid=None)`` and ``predict(rows)``. ``predict`` gives expected
daily counts (>= 0). Preprocessing is inside a scikit-learn pipeline that is fit on the training
rows only. Deep models are in ``deep.py`` and import PyTorch only when you use them.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class Forecaster:
    name = "base"
    space: dict[str, list[Any]] = {}  # search space for tuning; empty = nothing to tune

    def __init__(self, features: list[str], seed: int = 42, threads: int = 1, **params: Any):
        self.features = list(features)
        self.seed = seed
        self.threads = threads
        self.params = params

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "Forecaster":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Persistence(Forecaster):
    """The count of the origin day."""

    name = "persistence"

    def predict(self, rows):
        return np.expm1(rows["lag_0"].to_numpy(dtype=float))


class SeasonalNaive(Forecaster):
    """The count of the same calendar day one year before the target date."""

    name = "seasonal_naive"

    def predict(self, rows):
        v = rows["same_day_last_year"].fillna(rows["lag_0"]).to_numpy(dtype=float)
        return np.expm1(v)


class Climatology(Forecaster):
    """Mean count of the training rows per day of the year, smoothed over +/- 7 days."""

    name = "climatology"

    def fit(self, train, valid=None):
        doy = train["target_date"].dt.dayofyear.clip(upper=365).to_numpy()
        sums = np.bincount(doy - 1, weights=train["y"].to_numpy(dtype=float), minlength=365)
        cnts = np.bincount(doy - 1, minlength=365).astype(float)
        k = np.ones(15)
        ext_s = np.convolve(np.concatenate([sums[-7:], sums, sums[:7]]), k, mode="valid")
        ext_c = np.convolve(np.concatenate([cnts[-7:], cnts, cnts[:7]]), k, mode="valid")
        overall = train["y"].mean()
        self.table_ = np.where(ext_c > 0, ext_s / np.where(ext_c == 0, 1, ext_c), overall)
        return self

    def predict(self, rows):
        doy = rows["target_date"].dt.dayofyear.clip(upper=365).to_numpy()
        return self.table_[doy - 1]


class RidgeLog(Forecaster):
    """Ridge regression on log(1 + count) in a pipeline (impute, scale, ridge)."""

    name = "ridge"
    space = {"alpha": [0.1, 1.0, 10.0, 100.0]}

    def fit(self, train, valid=None):
        self.pipeline_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(self.params.get("alpha", 1.0)))),
        ]).fit(train[self.features], np.log1p(train["y"]))
        return self

    def predict(self, rows):
        return np.clip(np.expm1(self.pipeline_.predict(rows[self.features])), 0, None)


class GBMPoisson(Forecaster):
    """Gradient boosting with a Poisson loss on the raw counts."""

    name = "gbm"
    space = {
        "learning_rate": [0.03, 0.06, 0.1],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 1.0],
    }

    def fit(self, train, valid=None):
        p = {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0}
        p.update(self.params)
        est = HistGradientBoostingRegressor(loss="poisson", max_iter=250, random_state=self.seed, **p)
        with threadpool_limits(limits=self.threads):
            self.model_ = est.fit(train[self.features], train["y"])
        return self

    def predict(self, rows):
        with threadpool_limits(limits=self.threads):
            return self.model_.predict(rows[self.features])


def _lstm(**kw):
    from .deep import LSTMForecaster  # noqa: PLC0415  (imports torch)

    return LSTMForecaster(**kw)


MODELS: dict[str, Callable[..., Forecaster]] = {
    "persistence": Persistence,
    "seasonal_naive": SeasonalNaive,
    "climatology": Climatology,
    "ridge": RidgeLog,
    "gbm": GBMPoisson,
    "lstm": _lstm,
}
BASELINES = ("persistence", "seasonal_naive", "climatology")


def search_space(name: str) -> dict[str, list[Any]]:
    if name == "lstm":
        from .deep import LSTMForecaster  # noqa: PLC0415

        return LSTMForecaster.space
    return MODELS[name].space


def make_model(name: str, features: list[str], seed: int = 42, threads: int = 1, **params: Any) -> Forecaster:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; use one of {sorted(MODELS)}")
    return MODELS[name](features=features, seed=seed, threads=threads, **params)
