"""Problem 3 (zero days dropped) and problem 4 (non-vegetation and low-confidence detections)."""
import pandas as pd
import pytest

from firecast_id import synthetic
from firecast_id.climate import daily_known, load_climate
from firecast_id.firms import (DetectionSchemaError, FirmsApiSource, FolderSource, aggregate_daily,
                               clean_detections, load_series, validate_detections)


def _det(rows):
    return pd.DataFrame(rows, columns=["acq_date", "confidence", "type", "frp"])


def test_clean_keeps_only_vegetation_with_enough_confidence():
    df = _det([
        ("2020-01-01", 80, 0, 10.0),
        ("2020-01-01", 10, 0, 5.0),   # low confidence
        ("2020-01-01", 90, 2, 50.0),  # static land source (gas flare)
        ("2020-01-02", 95, 1, 80.0),  # volcano
        ("2020-01-02", 70, 3, 9.0),   # offshore
        ("bad date", 70, 0, 9.0),
    ])
    clean, rep = clean_detections(df, fire_types=(0,), min_confidence=30)
    assert len(clean) == 1
    assert rep.dropped_by_type == {1: 1, 2: 1, 3: 1}
    assert rep.dropped_low_confidence == 1 and rep.bad_rows == 1


def test_viirs_confidence_classes():
    v = validate_detections(_det([("2020-01-01", "l", 0, 1.0), ("2020-01-01", "n", 0, 1.0), ("2020-01-01", "h", 0, 1.0)]))
    assert v["confidence"].tolist() == [10, 50, 90]


def test_missing_columns_are_an_error():
    with pytest.raises(DetectionSchemaError):
        validate_detections(pd.DataFrame({"acq_date": ["2020-01-01"]}))


def test_aggregate_fills_zero_days():
    clean, _ = clean_detections(_det([("2020-01-01", 80, 0, 10.0), ("2020-01-04", 80, 0, 2.0), ("2020-01-04", 80, 0, 3.0)]))
    daily = aggregate_daily(clean)
    assert daily["count"].tolist() == [1, 0, 0, 2]
    assert daily["frp_sum"].tolist() == [10.0, 0.0, 0.0, 5.0]
    wide = aggregate_daily(clean, start="2019-12-30", end="2020-01-05")
    assert len(wide) == 7 and wide["count"].sum() == 3


def test_detections_round_trip_to_the_series():
    daily, _ = synthetic.generate("2019-08-01", "2019-08-20", seed=1)
    det = synthetic.detections_from_series(daily, seed=1)
    clean, rep = clean_detections(det)
    back = aggregate_daily(clean, "2019-08-01", "2019-08-20")
    assert back["count"].tolist() == daily["count"].tolist()
    assert sum(rep.dropped_by_type.values()) > 0 and rep.dropped_low_confidence > 0


def test_folder_source_filters_dates(tmp_path):
    _det([("2020-01-01", 80, 0, 1.0), ("2020-02-01", 80, 0, 1.0)]).to_csv(tmp_path / "a.csv", index=False)
    assert len(FolderSource(tmp_path).fetch("2020-01-01", "2020-01-31")) == 1
    with pytest.raises(FileNotFoundError):
        FolderSource(tmp_path / "none").fetch("2020-01-01", "2020-01-31")


def test_load_series_rejects_gaps(tmp_path, series):
    p = tmp_path / "s.csv"
    series.drop(series.index[5]).rename_axis("date").to_csv(p)
    with pytest.raises(DetectionSchemaError, match="calendar gaps"):
        load_series(p)
    series.rename_axis("date").to_csv(p)
    assert len(load_series(p)) == len(series)


def test_api_source_splits_requests_and_hides_the_key():
    calls = []

    def fake_open(url):
        calls.append(url)
        return "acq_date,confidence,type,frp\n2020-01-01,80,0,1.5\n"

    src = FirmsApiSource("SECRETKEY", opener=fake_open)
    df = src.fetch("2020-01-01", "2020-01-25")
    assert len(calls) == 3 and calls[0].endswith("/MODIS_SP/IDN/10/2020-01-01") and calls[2].endswith("/5/2020-01-21")
    assert len(df) == 3

    bad = FirmsApiSource("SECRETKEY", opener=lambda url: "Invalid MAP_KEY.")
    with pytest.raises(RuntimeError) as err:
        bad.fetch("2020-01-01", "2020-01-02")
    assert "SECRETKEY" not in str(err.value)
    with pytest.raises(ValueError):
        FirmsApiSource("")


def test_climate_values_wait_for_publication(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"month": ["2020-01", "2020-02"], "oni": [1.0, 2.0]}).to_csv(p, index=False)
    days = pd.date_range("2020-02-15", "2020-04-05")
    k = daily_known(load_climate(p), days, lag_days=30)
    assert pd.isna(k.loc["2020-02-29", "clim_oni"])  # January is not public yet
    assert k.loc["2020-03-01", "clim_oni"] == 1.0  # 31 Jan + 30 days
    assert k.loc["2020-03-29", "clim_oni"] == 1.0
    assert k.loc["2020-03-30", "clim_oni"] == 2.0  # 29 Feb + 30 days


def test_climate_file_checks(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"oni": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_climate(p)
