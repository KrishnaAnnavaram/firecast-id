"""FIRMS active-fire detections: read, validate, clean and aggregate to a daily series.

The daily series is on a complete calendar. A day with no kept detection has a count of 0.
Only the fire types in ``fire_types`` (default 0 = presumed vegetation fire) and detections
with a confidence at or above ``min_confidence`` are counted.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

REQUIRED = ("acq_date", "confidence", "type")
OPTIONAL = ("frp", "latitude", "longitude", "acq_time", "satellite", "instrument", "daynight")
# VIIRS gives a confidence class, MODIS gives 0..100. Map the classes to numbers.
CONFIDENCE_CLASS = {"l": 10, "low": 10, "n": 50, "nominal": 50, "h": 90, "high": 90}
TYPE_NAMES = {0: "vegetation fire", 1: "active volcano", 2: "other static land source", 3: "offshore"}


class DetectionSchemaError(ValueError):
    pass


@dataclass
class CleanReport:
    rows_in: int = 0
    rows_kept: int = 0
    bad_rows: int = 0
    dropped_by_type: dict[int, int] = field(default_factory=dict)
    dropped_low_confidence: int = 0

    def lines(self) -> list[str]:
        out = [f"rows in: {self.rows_in}", f"rows kept: {self.rows_kept}", f"bad rows: {self.bad_rows}",
               f"dropped low confidence: {self.dropped_low_confidence}"]
        for t, n in sorted(self.dropped_by_type.items()):
            out.append(f"dropped type {t} ({TYPE_NAMES.get(t, '?')}): {n}")
        return out


def _confidence(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    cls = col.astype(str).str.strip().str.lower().map(CONFIDENCE_CLASS)
    return num.fillna(cls)


def validate_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Check the columns and types. Return a frame with parsed dates and numeric fields."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DetectionSchemaError(f"missing required columns: {missing}")
    out = pd.DataFrame({
        "acq_date": pd.to_datetime(df["acq_date"], errors="coerce", format="mixed"),
        "confidence": _confidence(df["confidence"]),
        "type": pd.to_numeric(df["type"], errors="coerce"),
        "frp": pd.to_numeric(df["frp"], errors="coerce") if "frp" in df.columns else np.nan,
    })
    return out


def clean_detections(df: pd.DataFrame, fire_types=(0,), min_confidence: int = 30) -> tuple[pd.DataFrame, CleanReport]:
    v = validate_detections(df)
    rep = CleanReport(rows_in=len(v))
    bad = v["acq_date"].isna() | v["type"].isna() | v["confidence"].isna()
    rep.bad_rows = int(bad.sum())
    v = v[~bad]
    type_ok = v["type"].isin(list(fire_types))
    rep.dropped_by_type = {int(t): int(n) for t, n in v.loc[~type_ok, "type"].value_counts().items()}
    v = v[type_ok]
    conf_ok = v["confidence"] >= min_confidence
    rep.dropped_low_confidence = int((~conf_ok).sum())
    v = v[conf_ok]
    rep.rows_kept = len(v)
    return v.reset_index(drop=True), rep


def aggregate_daily(clean: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily ``count`` and ``frp_sum`` on a complete calendar. Days with no detection get 0."""
    if clean.empty and (start is None or end is None):
        raise ValueError("no detections are left and no start/end date is given")
    days = clean["acq_date"].dt.normalize()
    first = pd.Timestamp(start) if start else days.min()
    last = pd.Timestamp(end) if end else days.max()
    cal = pd.date_range(first, last, freq="D", name="date")
    g = clean.assign(day=days).groupby("day")
    out = pd.DataFrame({"count": g.size(), "frp_sum": g["frp"].sum(min_count=1)})
    out = out.reindex(cal)
    out["count"] = out["count"].fillna(0).astype(int)
    out["frp_sum"] = out["frp_sum"].fillna(0.0)
    return out


def read_folder(folder: str | Path) -> pd.DataFrame:
    """Read and join every ``*.csv`` file in a folder (for example the FIRMS yearly country files)."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {folder}; see data/README.md or run `firecast synth`")
    cols = list(REQUIRED + OPTIONAL)
    parts = [pd.read_csv(f, usecols=lambda c: c in cols) for f in files]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- sources


class DetectionSource(Protocol):
    def fetch(self, start: str, end: str) -> pd.DataFrame: ...


class FolderSource:
    """Detections from local CSV files (offline)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        df = read_folder(self.folder)
        d = pd.to_datetime(df["acq_date"], errors="coerce", format="mixed")
        return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)


class FirmsApiSource:
    """The FIRMS country API. It needs a free MAP_KEY. One request covers at most 10 days."""

    BASE = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
    MAX_DAYS = 10

    def __init__(self, map_key: str, source: str = "MODIS_SP", country: str = "IDN",
                 opener: Callable[[str], str] | None = None, timeout: float = 60.0):
        if not map_key:
            raise ValueError("FirmsApiSource needs FIRMS_MAP_KEY")
        self.map_key, self.source, self.country, self.timeout = map_key, source, country, timeout
        self._open = opener or self._http_get

    def _http_get(self, url: str) -> str:  # pragma: no cover - network
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def urls(self, start: str, end: str) -> list[str]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out = []
        while s <= e:
            n = min(self.MAX_DAYS, (e - s).days + 1)
            out.append(f"{self.BASE}/{self.map_key}/{self.source}/{self.country}/{n}/{s.date()}")
            s += pd.Timedelta(days=n)
        return out

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        parts = []
        for url in self.urls(start, end):
            text = self._open(url)
            if not text.strip() or text.lstrip().lower().startswith(("invalid", "error")):
                raise RuntimeError(f"FIRMS answered with an error for {url.replace(self.map_key, '***')}")
            parts.append(pd.read_csv(io.StringIO(text)))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(REQUIRED))


def load_series(path: str | Path) -> pd.DataFrame:
    """Read a daily series CSV (``date,count,frp_sum``) and check that the calendar is complete."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `firecast build-series` or `firecast synth`")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "count" not in df.columns:
        raise DetectionSchemaError("the series needs a `count` column")
    if df.index.duplicated().any():
        raise DetectionSchemaError("the series has duplicate dates")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    if len(full) != len(df):
        raise DetectionSchemaError(f"the series has {len(full) - len(df)} calendar gaps; rebuild it with zeros")
    if (df["count"] < 0).any():
        raise DetectionSchemaError("negative counts")
    if "frp_sum" not in df.columns:
        df["frp_sum"] = 0.0
    return df
