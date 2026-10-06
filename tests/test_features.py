"""Problem 1 (off-by-one evaluation) and problem 8 (windows cut at year starts)."""
import numpy as np
import pandas as pd
import pytest

from firecast_id.features import build_table, feature_names, windows


def test_windows_pair_each_input_with_its_own_target():
    v = np.arange(20, dtype=float)
    w = windows(v, n_lags=3, horizon=2)
    for i, o in enumerate(w.origin_pos):
        assert w.X[i].tolist() == [o - 2, o - 1, o]
        assert w.y[i] == o + 2
        assert w.target_pos[i] == o + 2
    assert w.origin_pos[-1] == 17  # last origin whose target (19) exists


def test_windows_include_future_rows():
    w = windows(np.arange(10.0), n_lags=2, horizon=3, include_future=True)
    assert w.origin_pos[-1] == 9
    assert np.isnan(w.y[-3:]).all()


@pytest.mark.parametrize("bad", [dict(n_lags=0, horizon=1), dict(n_lags=1, horizon=0), dict(n_lags=30, horizon=1)])
def test_windows_reject_bad_settings(bad):
    with pytest.raises(ValueError):
        windows(np.arange(10.0), **bad)


def test_table_target_is_the_count_of_the_target_date(series):
    t = build_table(series, horizon=7, n_lags=5)
    assert ((t["target_date"] - t["origin"]).dt.days == 7).all()
    assert np.array_equal(t["y"].to_numpy(), series.loc[t["target_date"], "count"].to_numpy(dtype=float))
    assert np.allclose(t["lag_0"], np.log1p(series.loc[t["origin"], "count"].to_numpy()))


def test_features_do_not_read_after_the_origin(series):
    o = pd.Timestamp("2018-06-30")
    changed = series.copy()
    changed.loc[changed.index > o, ["count", "frp_sum"]] = 99999
    a = build_table(series, 14, 10).set_index("origin").loc[o]
    b = build_table(changed, 14, 10).set_index("origin").loc[o]
    feats = feature_names(10, [])
    pd.testing.assert_series_equal(a[feats], b[feats])
    assert a["y"] != b["y"]


def test_first_days_of_a_year_have_rows_with_december_origins(series):
    t = build_table(series, 7, 14)
    jan = t[t["target_date"] == pd.Timestamp("2019-01-03")]
    assert len(jan) == 1 and jan["origin"].iloc[0] == pd.Timestamp("2018-12-27")


def test_same_day_last_year(series):
    t = build_table(series, 1, 3).set_index("target_date")
    d = pd.Timestamp("2019-03-10")
    assert np.isclose(t.loc[d, "same_day_last_year"], np.log1p(series.loc[d - pd.Timedelta(days=365), "count"]))


def test_climate_columns_are_read_at_the_origin(series):
    s = series.copy()
    s["clim_x"] = np.arange(len(s), dtype=float)
    t = build_table(s, 30, 3)
    assert np.array_equal(t["clim_x"].to_numpy(), s.index.get_indexer(t["origin"]).astype(float))
