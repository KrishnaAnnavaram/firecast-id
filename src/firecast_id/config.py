"""Settings from environment variables and a local ``.env`` file. Every value has an offline default."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read ``KEY=VALUE`` lines. Empty values are skipped. Real environment variables win."""
    path = Path(path)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = Path("data/modis_indonesia")
    series_path: Path = Path("data/daily_series.csv")
    climate_path: Path | None = None
    output_dir: Path = Path("reports")
    fire_types: tuple[int, ...] = (0,)
    min_confidence: int = 30
    horizons: tuple[int, ...] = (1, 7, 14, 30)
    n_lags: int = 14
    peak_months: tuple[int, ...] = (8, 9, 10)
    climate_lag_days: int = 30
    seed: int = 42
    tuning_trials: int = 6
    threads: int = 1
    firms_map_key: str | None = None
    firms_source: str = "MODIS_SP"
    firms_country: str = "IDN"

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1 or max(self.horizons) > 365:
            raise ValueError("horizons must be between 1 and 365 days")
        if not 1 <= self.n_lags <= 120:
            raise ValueError("n_lags must be between 1 and 120")
        if not 0 <= self.min_confidence <= 100:
            raise ValueError("min_confidence must be between 0 and 100")
        if any(t not in (0, 1, 2, 3) for t in self.fire_types):
            raise ValueError("fire types are 0, 1, 2 or 3")
        if any(not 1 <= m <= 12 for m in self.peak_months):
            raise ValueError("peak months are 1..12")
        if self.tuning_trials < 1 or self.threads < 1:
            raise ValueError("tuning_trials and threads must be at least 1")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = {**read_dotenv(), **os.environ} if env is None else env
        d = cls()
        return cls(
            raw_dir=Path(e.get("FIRECAST_RAW_DIR", str(d.raw_dir))),
            series_path=Path(e.get("FIRECAST_SERIES", str(d.series_path))),
            climate_path=Path(e["FIRECAST_CLIMATE"]) if e.get("FIRECAST_CLIMATE") else None,
            output_dir=Path(e.get("FIRECAST_OUTPUT_DIR", str(d.output_dir))),
            fire_types=_ints(e["FIRECAST_FIRE_TYPES"]) if e.get("FIRECAST_FIRE_TYPES") else d.fire_types,
            min_confidence=int(e.get("FIRECAST_MIN_CONFIDENCE", d.min_confidence)),
            horizons=_ints(e["FIRECAST_HORIZONS"]) if e.get("FIRECAST_HORIZONS") else d.horizons,
            n_lags=int(e.get("FIRECAST_N_LAGS", d.n_lags)),
            peak_months=_ints(e["FIRECAST_PEAK_MONTHS"]) if e.get("FIRECAST_PEAK_MONTHS") else d.peak_months,
            climate_lag_days=int(e.get("FIRECAST_CLIMATE_LAG_DAYS", d.climate_lag_days)),
            seed=int(e.get("FIRECAST_SEED", d.seed)),
            tuning_trials=int(e.get("FIRECAST_TUNING_TRIALS", d.tuning_trials)),
            threads=int(e.get("FIRECAST_THREADS", d.threads)),
            firms_map_key=e.get("FIRMS_MAP_KEY") or None,
            firms_source=e.get("FIRMS_SOURCE", d.firms_source),
            firms_country=e.get("FIRMS_COUNTRY", d.firms_country),
        )
