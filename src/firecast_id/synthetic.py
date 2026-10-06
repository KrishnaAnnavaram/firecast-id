"""Synthetic fire data with a known process, for the demo and the tests (no download).

* A monthly ENSO-like index (AR(1)) and a monthly rainfall value.
* A daily fire intensity with a dry-season peak (August to October), a drought effect from the
  index two months earlier, and a persistent daily latent state.
* Negative-binomial daily counts (many low days, a few very high days).
* Optional detection rows in the FIRMS layout, with non-vegetation types and low-confidence rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(start: str = "2008-01-01", end: str = "2023-12-31", seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(daily, climate)``. ``daily`` has ``count`` and ``frp_sum``; ``climate`` is monthly."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D", name="date")
    months = pd.date_range(days[0].replace(day=1), days[-1], freq="MS")

    oni = np.zeros(len(months))
    for i in range(1, len(months)):
        oni[i] = 0.92 * oni[i - 1] + rng.normal(0, 0.28)
    mo = months.month.to_numpy()
    rain = np.clip(220 + 120 * np.cos(2 * np.pi * (mo - 1) / 12) - 40 * oni + rng.normal(0, 25, len(months)), 5, None)
    climate = pd.DataFrame({"month": months.strftime("%Y-%m"), "oni": oni.round(3), "rain_mm": rain.round(1)})

    # drought driver: the index two months earlier, on each day
    lagged = pd.Series(oni, index=months).shift(2).bfill().fillna(0.0)
    drought = lagged.reindex(days, method="ffill").to_numpy()
    doy = days.dayofyear.to_numpy()
    season = 1.6 * np.exp(-0.5 * ((doy - 258) / 32.0) ** 2)  # peak in mid September
    latent = np.zeros(len(days))
    for i in range(1, len(days)):
        latent[i] = 0.85 * latent[i - 1] + rng.normal(0, 0.25)
    log_mu = 2.6 + 2.2 * season + 0.9 * np.clip(drought, -1, None) * season + latent
    mu = np.exp(log_mu)
    shape = 2.0
    counts = rng.poisson(rng.gamma(shape, mu / shape))
    frp = np.array([rng.gamma(1.5, 25.0, c).sum() if c else 0.0 for c in counts])
    daily = pd.DataFrame({"count": counts.astype(int), "frp_sum": frp.round(1)}, index=days)
    return daily, climate


def detections_from_series(daily: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Detection rows in the FIRMS layout. The kept vegetation rows give back ``daily['count']``.

    Each day also gets non-vegetation rows (types 1-3) and low-confidence vegetation rows, which
    the cleaning step must remove.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day, n in daily["count"].items():
        n_static = rng.poisson(3)
        n_volcano = rng.poisson(0.3)
        n_offshore = rng.poisson(0.5)
        n_low = rng.poisson(0.15 * n + 1)
        kinds = [(0, n, (30, 101)), (0, n_low, (0, 30)), (2, n_static, (50, 101)),
                 (1, n_volcano, (50, 101)), (3, n_offshore, (50, 101))]
        for typ, k, (lo, hi) in kinds:
            for _ in range(k):
                rows.append((day.strftime("%Y-%m-%d"), int(rng.integers(lo, hi)), typ,
                             round(float(rng.gamma(1.5, 25.0)), 1),
                             round(float(rng.uniform(-8, 4)), 4), round(float(rng.uniform(95, 140)), 4)))
    return pd.DataFrame(rows, columns=["acq_date", "confidence", "type", "frp", "latitude", "longitude"])
