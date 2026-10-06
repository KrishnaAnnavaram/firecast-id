"""Year-by-year backtest with a separate validation year.

For each test year Y and each horizon h:

1. Tuning: fit on target dates before Y-1, score on Y-1 (``tuning.py``).
2. Refit with the best parameters on all target dates before Y.
3. Forecast every target date in Y. The origin of a target in early January is in December of
   Y-1, so every day of Y is forecast.

All predictions go into one table with the target date. Every metric and test reads that table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .features import build_table, feature_names
from .metrics import diebold_mariano, mae, mase, mase_scale, poisson_deviance, rmse
from .models import BASELINES, make_model
from .tuning import tune


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    dm_tests: pd.DataFrame
    tuning: list[dict[str, Any]] = field(default_factory=list)
    features: list[str] = field(default_factory=list)


def _score(g: pd.DataFrame, scale: float, peak_months: tuple[int, ...]) -> dict[str, float]:
    peak = g["target_date"].dt.month.isin(peak_months)
    return {
        "n": int(len(g)),
        "mae": mae(g["y"], g["pred"]),
        "rmse": rmse(g["y"], g["pred"]),
        "mase": mase(g["y"], g["pred"], scale),
        "peak_mae": mae(g.loc[peak, "y"], g.loc[peak, "pred"]) if peak.any() else float("nan"),
        "poisson_deviance": poisson_deviance(g["y"], g["pred"]),
    }


def check_years(series: pd.DataFrame, test_years: list[int]) -> None:
    first, last = series.index.min(), series.index.max()
    for y in test_years:
        if pd.Timestamp(f"{y}-12-31") > last:
            raise ValueError(f"test year {y} is not complete in the series (last date {last.date()})")
        if pd.Timestamp(f"{y - 2}-01-01") < first:
            raise ValueError(f"test year {y} needs at least two earlier years (training and validation)")


def run_backtest(series: pd.DataFrame, models: list[str], settings: Settings, test_years: list[int],
                 do_tune: bool = True, reference: str | None = None) -> BacktestResult:
    check_years(series, test_years)
    clim = [c for c in series.columns if c.startswith("clim_")]
    feats = feature_names(settings.n_lags, clim)
    preds, tuning_log = [], []
    for h in settings.horizons:
        table = build_table(series, h, settings.n_lags)
        table = table[table["y"].notna()]
        for year in test_years:
            start = pd.Timestamp(f"{year}-01-01")
            valid_start = pd.Timestamp(f"{year - 1}-01-01")
            train = table[table["target_date"] < start]
            test = table[table["target_date"].dt.year == year]
            for name in models:
                params: dict[str, Any] = {}
                if do_tune:
                    inner = table[table["target_date"] < valid_start]
                    valid = table[table["target_date"].dt.year == year - 1]
                    tr = tune(name, feats, inner, valid, settings.tuning_trials, settings.seed, settings.threads)
                    params = tr.best_params
                    if tr.trials:
                        tuning_log.append({"model": name, "horizon": h, "test_year": year,
                                           "best_params": tr.best_params, "best_valid_mae": tr.best_valid_mae,
                                           "trials": tr.trials})
                m = make_model(name, feats, seed=settings.seed, threads=settings.threads, **params).fit(train)
                out = test[["origin", "target_date", "y"]].copy()
                out["pred"] = np.clip(m.predict(test), 0, None)
                out["model"], out["horizon"], out["test_year"] = name, h, year
                preds.append(out)
    pred = pd.concat(preds, ignore_index=True)

    rows = []
    for (name, h, year), g in pred.groupby(["model", "horizon", "test_year"], sort=False):
        scale = mase_scale(series.loc[: f"{year - 1}-12-31", "count"])
        rows.append({"model": name, "horizon": h, "test_year": str(year), **_score(g, scale, settings.peak_months)})
    for (name, h), g in pred.groupby(["model", "horizon"], sort=False):
        scale = mase_scale(series.loc[: f"{min(test_years) - 1}-12-31", "count"])
        rows.append({"model": name, "horizon": h, "test_year": "all", **_score(g, scale, settings.peak_months)})
    metrics = pd.DataFrame(rows)

    ref = reference or next((b for b in BASELINES if b in models), None)
    dm_rows = []
    if ref is not None:
        for h in settings.horizons:
            base = pred[(pred["model"] == ref) & (pred["horizon"] == h)].sort_values("target_date")
            for name in models:
                if name == ref:
                    continue
                other = pred[(pred["model"] == name) & (pred["horizon"] == h)].sort_values("target_date")
                if not np.array_equal(other["target_date"].to_numpy(), base["target_date"].to_numpy()):
                    raise RuntimeError("the prediction tables are not aligned by target date")
                r = diebold_mariano(other["y"] - other["pred"], base["y"] - base["pred"], horizon=h)
                dm_rows.append({"model": name, "reference": ref, "horizon": h, "dm_stat": r.statistic,
                                "p_value": r.p_value, "mean_abs_error_diff": r.mean_diff, "n": r.n})
    return BacktestResult(pred, metrics, pd.DataFrame(dm_rows), tuning_log, feats)
