"""Command line: ``firecast <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__, synthetic
from .backtest import run_backtest
from .config import Settings
from .firms import (DetectionSchemaError, FirmsApiSource, FolderSource, aggregate_daily, clean_detections,
                    load_series)
from .models import BASELINES, MODELS
from .pipeline import final_forecast, model_input

DEFAULT_MODELS = "persistence,seasonal_naive,climatology,ridge,gbm"


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _input(args, s: Settings) -> pd.DataFrame:
    series = load_series(args.series or s.series_path)
    climate = args.climate if args.climate is not None else s.climate_path
    return model_input(series, climate or None, s.climate_lag_days)


def cmd_synth(args, s):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily, climate = synthetic.generate(args.start, args.end, seed=s.seed)
    daily.rename_axis("date").to_csv(out / "daily_series.csv")
    climate.to_csv(out / "climate.csv", index=False)
    print(f"wrote {len(daily)} days to {out / 'daily_series.csv'} and {len(climate)} months to {out / 'climate.csv'}")
    if args.detections_from:
        raw = out / "raw"
        raw.mkdir(exist_ok=True)
        part = daily[daily.index.year >= args.detections_from]
        det = synthetic.detections_from_series(part, seed=s.seed)
        for year, g in det.groupby(det["acq_date"].str[:4]):
            g.to_csv(raw / f"synthetic_{year}.csv", index=False)
        print(f"wrote {len(det)} synthetic detections to {raw}")


def cmd_build_series(args, s):
    src = FolderSource(args.raw_dir or s.raw_dir)
    raw = src.fetch(args.start or "1900-01-01", args.end or "2100-12-31")
    clean, rep = clean_detections(raw, fire_types=s.fire_types, min_confidence=s.min_confidence)
    print("\n".join(rep.lines()))
    daily = aggregate_daily(clean, args.start, args.end)
    out = Path(args.out or s.series_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.rename_axis("date").to_csv(out)
    print(f"wrote {len(daily)} days ({int((daily['count'] == 0).sum())} zero days) to {out}")


def cmd_fetch(args, s):
    if not s.firms_map_key:
        raise ValueError("set FIRMS_MAP_KEY (free key from the FIRMS website) to use `firecast fetch`")
    src = FirmsApiSource(s.firms_map_key, s.firms_source, s.firms_country)
    df = src.fetch(args.start, args.end)
    out = Path(args.out_dir or s.raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"firms_{s.firms_source}_{s.firms_country}_{args.start}_{args.end}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} detections to {path}")


def cmd_backtest(args, s):
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in names if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; use {sorted(MODELS)}")
    if not any(m in BASELINES for m in names):
        print("note: no baseline in --models; the Diebold-Mariano tests need one")
    years = [int(y) for y in args.test_years.split(",")]
    res = run_backtest(_input(args, s), names, s, years, do_tune=not args.no_tune)
    print(f"features ({len(res.features)}): {', '.join(res.features)}")
    print("\n== all test years ==")
    _print(res.metrics[res.metrics["test_year"] == "all"].sort_values(["horizon", "mae"]))
    if not res.dm_tests.empty:
        print(f"\n== Diebold-Mariano tests (absolute error) against {res.dm_tests['reference'].iloc[0]} ==")
        _print(res.dm_tests)
    out = Path(args.out) if args.out else s.output_dir / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    res.predictions.to_csv(out / "predictions.csv", index=False)
    res.metrics.to_csv(out / "metrics.csv", index=False)
    res.dm_tests.to_csv(out / "dm_tests.csv", index=False)
    run = {"models": names, "test_years": years, "tuned": not args.no_tune, "features": res.features,
           "settings": {"horizons": s.horizons, "n_lags": s.n_lags, "seed": s.seed, "fire_types": s.fire_types,
                        "min_confidence": s.min_confidence, "tuning_trials": s.tuning_trials,
                        "peak_months": s.peak_months, "climate_lag_days": s.climate_lag_days},
           "tuning": res.tuning}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote reports to {out}")


def cmd_forecast(args, s):
    fc = final_forecast(_input(args, s), args.model, s, do_tune=not args.no_tune)
    _print(fc)


def cmd_demo(args, s):
    out = Path(args.out_dir) if args.out_dir else s.output_dir / "demo"
    cmd_synth(argparse.Namespace(out_dir=out, start="2012-01-01", end="2023-12-31", detections_from=None), s)
    ns = argparse.Namespace(series=out / "daily_series.csv", climate=out / "climate.csv", models=DEFAULT_MODELS,
                            test_years="2021,2022,2023", no_tune=False, out=out / "backtest")
    cmd_backtest(ns, s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="firecast", description="Daily fire hotspot forecasts for Indonesia.")
    p.add_argument("--version", action="version", version=f"firecast-id {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a synthetic daily series and climate file")
    s.add_argument("--out-dir", default="data/synthetic")
    s.add_argument("--start", default="2008-01-01")
    s.add_argument("--end", default="2023-12-31")
    s.add_argument("--detections-from", type=int, help="also write detection rows from this year on")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("build-series", help="clean FIRMS detections and write the daily series")
    s.add_argument("--raw-dir")
    s.add_argument("--out")
    s.add_argument("--start")
    s.add_argument("--end")
    s.set_defaults(func=cmd_build_series)

    s = sub.add_parser("fetch", help="download detections from the FIRMS API (needs FIRMS_MAP_KEY)")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out-dir")
    s.set_defaults(func=cmd_fetch)

    for name, fn, hlp in (("backtest", cmd_backtest, "year-by-year backtest with a validation year"),
                          ("forecast", cmd_forecast, "forecast the next days with a final model")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--series")
        s.add_argument("--climate", help="monthly climate CSV (default FIRECAST_CLIMATE)")
        s.add_argument("--no-tune", action="store_true", help="use default parameters, no search")
        if name == "backtest":
            s.add_argument("--models", default=DEFAULT_MODELS)
            s.add_argument("--test-years", default="2021,2022,2023")
            s.add_argument("--out")
        else:
            s.add_argument("--model", default="gbm", choices=sorted(MODELS))
        s.set_defaults(func=fn)

    s = sub.add_parser("demo", help="offline demo on synthetic data")
    s.add_argument("--out-dir")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args, Settings.from_env())
    except (DetectionSchemaError, FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
