"""Glue: build the model input from the daily series and the climate file, and make the final forecast."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .climate import daily_known, load_climate
from .config import Settings
from .features import build_table, feature_names
from .models import make_model
from .tuning import tune


def model_input(series: pd.DataFrame, climate_path: str | Path | None, lag_days: int) -> pd.DataFrame:
    """The daily series plus the climate values that were public on each day (``clim_*`` columns)."""
    out = series[["count", "frp_sum"]].copy()
    if climate_path:
        known = daily_known(load_climate(climate_path), out.index, lag_days=lag_days)
        out = out.join(known)
    return out


def final_forecast(series: pd.DataFrame, model: str, settings: Settings, do_tune: bool = True) -> pd.DataFrame:
    """Fit a new model per horizon on all rows and forecast ``last date + h`` for each horizon.

    With ``do_tune`` the last complete year is the validation year for the search, and the final
    fit uses all rows with the best parameters.
    """
    clim = [c for c in series.columns if c.startswith("clim_")]
    feats = feature_names(settings.n_lags, clim)
    last = series.index.max()
    rows = []
    for h in settings.horizons:
        table = build_table(series, h, settings.n_lags, include_future=True)
        known = table[table["y"].notna()]
        params = {}
        if do_tune:
            vyear = known["target_date"].dt.year.max() - 1
            inner = known[known["target_date"].dt.year < vyear]
            valid = known[known["target_date"].dt.year == vyear]
            if len(inner) and len(valid):
                params = tune(model, feats, inner, valid, settings.tuning_trials, settings.seed, settings.threads).best_params
        m = make_model(model, feats, seed=settings.seed, threads=settings.threads, **params).fit(known)
        row = table[table["origin"] == last]
        rows.append({"horizon": h, "origin": last.date(), "target_date": row["target_date"].iloc[0].date(),
                     "forecast": float(np.clip(m.predict(row), 0, None)[0]), "params": params})
    return pd.DataFrame(rows)
