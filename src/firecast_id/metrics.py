"""Error metrics and the Diebold-Mariano test.

All functions take arrays that are already aligned by target date. ``backtest.py`` builds
them from one table, keyed by model, horizon and target date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mase_scale(train_y) -> float:
    """Mean absolute one-day change of the training series (the in-sample persistence error)."""
    d = np.abs(np.diff(np.asarray(train_y, float)))
    s = float(d.mean()) if len(d) else float("nan")
    return s if s > 0 else float("nan")


def mase(y, p, scale: float) -> float:
    return mae(y, p) / scale


def poisson_deviance(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0.0)
    return float(2 * np.mean(term - (y - p)))


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float  # mean loss of model A minus mean loss of model B (negative = A is better)
    n: int


def diebold_mariano(e_a, e_b, horizon: int = 1, loss: str = "abs") -> DMResult:
    """Two-sided Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    ``e_a`` and ``e_b`` are forecast errors of two models on the same target dates, in date order.
    The variance uses a Newey-West (Bartlett) estimate with ``horizon - 1`` lags.
    """
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    if e_a.shape != e_b.shape or e_a.ndim != 1:
        raise ValueError("the two error arrays must be 1-D and have the same length")
    if loss == "abs":
        d = np.abs(e_a) - np.abs(e_b)
    elif loss == "squared":
        d = e_a**2 - e_b**2
    else:
        raise ValueError("loss is 'abs' or 'squared'")
    n = len(d)
    if n < 10:
        raise ValueError("at least 10 paired errors are necessary")
    dbar = d.mean()
    dc = d - dbar
    lags = max(0, horizon - 1)
    var = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return DMResult(statistic=0.0, p_value=1.0, mean_diff=float(dbar), n=n)
    dm = dbar / np.sqrt(var / n)
    h = horizon
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * corr)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(statistic=stat, p_value=p, mean_diff=float(dbar), n=n)
