"""Monthly climate drivers (for example ONI for ENSO, DMI for the Indian Ocean Dipole, rainfall).

A monthly value is known only after the month ends and the agency publishes it. ``daily_known``
gives, for each day, the latest monthly value that was public on that day:
``month end + lag_days <= day``. Features read this table at the origin, so no future
climate value goes into a forecast.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_climate(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a ``month`` column (``YYYY-MM``) and one or more numeric driver columns."""
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError("the climate file needs a `month` column (YYYY-MM)")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    if df["month"].isna().any():
        raise ValueError("unreadable months in the climate file")
    drivers = [c for c in df.columns if c != "month"]
    if not drivers:
        raise ValueError("the climate file has no driver columns")
    for c in drivers:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df.sort_values("month").reset_index(drop=True)


def daily_known(monthly: pd.DataFrame, dates: pd.DatetimeIndex, lag_days: int = 30) -> pd.DataFrame:
    """For each date, the latest monthly values that were public on that date (NaN before the first)."""
    m = monthly.copy()
    m["available"] = m["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")
    drivers = [c for c in m.columns if c not in ("month", "available")]
    merged = pd.merge_asof(left, m[["available", *drivers]].sort_values("available"),
                           left_on="date", right_on="available", direction="backward")
    out = merged.set_index("date")[drivers]
    out.columns = [f"clim_{c}" for c in drivers]
    return out.reindex(pd.DatetimeIndex(dates))
