"""Windows and features for a direct forecast at one horizon.

``windows`` is the ONE function that pairs an input window with its target. It returns the
origin date and the target date of each row, so every metric compares a forecast with the
value of its own target date.

A feature row for origin ``o`` and horizon ``h``:

* reads counts, FRP and climate values on days ``<= o`` only;
* has calendar features of the target date ``o + h`` (known in advance);
* has the target ``y = count[o + h]`` (NaN after the end of the series).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Windows:
    X: np.ndarray  # (n_rows, n_lags), oldest lag first, last column = value at the origin
    y: np.ndarray  # (n_rows,), value at origin + horizon (NaN after the end)
    origin_pos: np.ndarray
    target_pos: np.ndarray


def windows(values: np.ndarray, n_lags: int, horizon: int, include_future: bool = False) -> Windows:
    """Sliding windows over a 1-D array.

    Row ``i`` has the inputs ``values[o - n_lags + 1 .. o]`` and the target ``values[o + horizon]``.
    With ``include_future`` the last origins (whose target is after the end) are kept with y = NaN.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n_lags < 1 or horizon < 1:
        raise ValueError("n_lags and horizon must be at least 1")
    last = n - 1 if include_future else n - 1 - horizon
    origins = np.arange(n_lags - 1, last + 1)
    if len(origins) == 0:
        raise ValueError("the series is too short for these windows")
    idx = origins[:, None] + np.arange(-n_lags + 1, 1)[None, :]
    X = values[idx]
    t = origins + horizon
    y = np.full(len(origins), np.nan)
    ok = t < n
    y[ok] = values[t[ok]]
    return Windows(X=X, y=y, origin_pos=origins, target_pos=t)


def _at(a: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(pos.shape, np.nan)
    ok = (pos >= 0) & (pos < len(a))
    out[ok] = a[pos[ok]]
    return out


def feature_names(n_lags: int, climate_cols: list[str]) -> list[str]:
    return [f"lag_{k}" for k in range(n_lags)] + [
        "mean_7", "mean_30", "max_30", "frp_mean_7", "same_day_last_year", "doy_sin", "doy_cos", *climate_cols,
    ]


def build_table(series: pd.DataFrame, horizon: int, n_lags: int, include_future: bool = False) -> pd.DataFrame:
    """Feature rows for every origin. ``series`` has a complete daily index and a ``count`` column.

    Optional columns: ``frp_sum`` and ``clim_*`` (climate values already known on each day).
    """
    dates = pd.DatetimeIndex(series.index)
    logc = np.log1p(series["count"].to_numpy(dtype=float))
    w = windows(logc, n_lags, horizon, include_future=include_future)
    o, t = w.origin_pos, w.target_pos
    s = pd.Series(logc)
    frp = np.log1p(series["frp_sum"].to_numpy(dtype=float)) if "frp_sum" in series else np.zeros(len(logc))
    clim_cols = [c for c in series.columns if c.startswith("clim_")]

    ext = pd.date_range(dates[0], periods=len(dates) + horizon, freq="D")
    tdates = ext[t]
    doy = tdates.dayofyear.to_numpy()
    table = {
        "origin": dates[o],
        "target_date": tdates,
        "y": np.expm1(w.y),
    }
    for k in range(n_lags):
        table[f"lag_{k}"] = w.X[:, n_lags - 1 - k]  # lag_0 = value at the origin
    table["mean_7"] = s.rolling(7, min_periods=1).mean().to_numpy()[o]
    table["mean_30"] = s.rolling(30, min_periods=1).mean().to_numpy()[o]
    table["max_30"] = s.rolling(30, min_periods=1).max().to_numpy()[o]
    table["frp_mean_7"] = pd.Series(frp).rolling(7, min_periods=1).mean().to_numpy()[o]
    base = t - 365  # same calendar day one year before the target; it is before the origin when h <= 365
    sdly = _at(logc, base)
    sdly[base > o] = np.nan
    table["same_day_last_year"] = sdly
    table["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    table["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in clim_cols:
        table[c] = series[c].to_numpy(dtype=float)[o]
    out = pd.DataFrame(table)
    out["y"] = out["y"].round().astype("float")
    return out
