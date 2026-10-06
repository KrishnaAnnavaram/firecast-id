import pandas as pd
import pytest

from firecast_id.cli import main
from firecast_id.config import Settings, read_dotenv


def test_settings_from_env():
    s = Settings.from_env({"FIRECAST_HORIZONS": "1,3", "FIRECAST_FIRE_TYPES": "0,2", "FIRECAST_MIN_CONFIDENCE": "50",
                           "FIRMS_MAP_KEY": "abc", "FIRECAST_CLIMATE": "c.csv"})
    assert s.horizons == (1, 3) and s.fire_types == (0, 2) and s.min_confidence == 50
    assert s.firms_map_key == "abc" and str(s.climate_path) == "c.csv"


@pytest.mark.parametrize("env", [{"FIRECAST_HORIZONS": "0"}, {"FIRECAST_FIRE_TYPES": "7"},
                                 {"FIRECAST_MIN_CONFIDENCE": "101"}, {"FIRECAST_PEAK_MONTHS": "13"}])
def test_bad_settings(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FIRECAST_N_LAGS=9\nFIRMS_MAP_KEY=\n", encoding="utf-8")
    assert read_dotenv(tmp_path / ".env") == {"FIRECAST_N_LAGS": "9"}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIRECAST_N_LAGS", raising=False)
    assert Settings.from_env().n_lags == 9


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FIRECAST_HORIZONS", "1,7")
    monkeypatch.setenv("FIRECAST_TUNING_TRIALS", "1")
    d = tmp_path / "syn"
    assert main(["synth", "--out-dir", str(d), "--start", "2016-01-01", "--end", "2019-12-31",
                 "--detections-from", "2019"]) == 0
    series = tmp_path / "rebuilt.csv"
    assert main(["build-series", "--raw-dir", str(d / "raw"), "--out", str(series)]) == 0
    out = capsys.readouterr().out
    assert "dropped type 2" in out and "dropped low confidence" in out
    rebuilt = pd.read_csv(series)
    orig = pd.read_csv(d / "daily_series.csv")
    assert rebuilt["count"].tolist() == orig[orig["date"] >= "2019-01-01"]["count"].tolist()

    rep = tmp_path / "bt"
    assert main(["backtest", "--series", str(d / "daily_series.csv"), "--climate", str(d / "climate.csv"),
                 "--models", "persistence,ridge", "--test-years", "2019", "--out", str(rep)]) == 0
    assert (rep / "metrics.csv").exists() and (rep / "run.json").exists() and (rep / "dm_tests.csv").exists()
    assert "clim_oni" in capsys.readouterr().out
    assert main(["forecast", "--series", str(d / "daily_series.csv"), "--model", "ridge", "--no-tune"]) == 0


def test_cli_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["fetch", "--start", "2020-01-01", "--end", "2020-01-02"]) == 2
    assert "FIRMS_MAP_KEY" in capsys.readouterr().err
    assert main(["backtest", "--series", str(tmp_path / "none.csv")]) == 2
