"""Problem 2 (tuning on training data), 5 (untuned final model), 7 (no baselines), 8 (test coverage), 9 (seeds)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from firecast_id import backtest as bt
from firecast_id import tuning
from firecast_id.features import build_table, feature_names
from firecast_id.models import BASELINES, make_model
from firecast_id.pipeline import final_forecast


@pytest.fixture(scope="module")
def table(series):
    return build_table(series, 7, 7).dropna(subset=["y"])


FEATS = feature_names(7, [])


@pytest.mark.parametrize("name", ["persistence", "seasonal_naive", "climatology", "ridge", "gbm"])
def test_models_fit_and_predict_counts(name, table):
    train, test = table.iloc[:-100], table.iloc[-100:]
    pred = make_model(name, FEATS, seed=1).fit(train).predict(test)
    assert pred.shape == (100,) and (pred >= 0).all() and np.isfinite(pred).all()


def test_baselines_are_simple_functions(table):
    row = table.iloc[[500]]
    assert make_model("persistence", FEATS).predict(row)[0] == pytest.approx(np.expm1(row["lag_0"].iloc[0]))
    assert make_model("seasonal_naive", FEATS).predict(row)[0] == pytest.approx(np.expm1(row["same_day_last_year"].iloc[0]))
    assert set(BASELINES) == {"persistence", "seasonal_naive", "climatology"}


def test_gbm_is_deterministic(table):
    a = make_model("gbm", FEATS, seed=4).fit(table).predict(table.tail(30))
    b = make_model("gbm", FEATS, seed=4).fit(table).predict(table.tail(30))
    assert np.array_equal(a, b)


def test_ridge_scaler_sees_training_rows_only(table):
    train = table[table["target_date"].dt.year <= 2017]
    m = make_model("ridge", FEATS).fit(train)
    imputed = m.pipeline_.named_steps["impute"].transform(train[FEATS])
    assert np.allclose(m.pipeline_.named_steps["scale"].mean_, imputed.mean(axis=0))


def test_candidates_are_seeded_and_bounded():
    space = {"a": [1, 2, 3], "b": [1, 2]}
    assert tuning.candidates(space, 2, 0) == tuning.candidates(space, 2, 0)
    assert len(tuning.candidates(space, 2, 0)) == 2
    assert len(tuning.candidates(space, 10, 0)) == 6
    assert tuning.candidates({}, 5, 0) == []


def test_tuning_never_reads_the_test_year(series, settings, monkeypatch):
    seen = []
    real = tuning.tune

    def spy(name, feats, train, valid, *a, **k):
        seen.append((train["target_date"].max(), valid["target_date"].min(), valid["target_date"].max()))
        return real(name, feats, train, valid, *a, **k)

    monkeypatch.setattr(bt, "tune", spy)
    bt.run_backtest(series, ["persistence", "ridge"], settings, [2019], do_tune=True)
    assert seen
    for train_max, vmin, vmax in seen:
        assert train_max < pd.Timestamp("2018-01-01")
        assert vmin >= pd.Timestamp("2018-01-01") and vmax <= pd.Timestamp("2018-12-31")


def test_refit_uses_the_tuned_parameters(series, settings, monkeypatch):
    made = []
    real = bt.make_model

    def spy(name, feats, **kw):
        made.append((name, {k: v for k, v in kw.items() if k not in ("seed", "threads")}))
        return real(name, feats, **kw)

    monkeypatch.setattr(bt, "make_model", spy)
    res = bt.run_backtest(series, ["ridge"], settings, [2019], do_tune=True)
    best = {(t["horizon"]): t["best_params"] for t in res.tuning}
    assert [p for _, p in made] == [best[1], best[7]]
    assert all("alpha" in p for _, p in made)


def test_backtest_forecasts_every_day_of_the_test_year(series, settings):
    res = bt.run_backtest(series, ["persistence", "climatology", "gbm"], settings, [2018, 2019], do_tune=False)
    counts = res.predictions.groupby(["model", "horizon", "test_year"]).size()
    assert (counts == 365).all()
    first = res.predictions[(res.predictions["test_year"] == 2019) & (res.predictions["horizon"] == 7)]
    assert first["target_date"].min() == pd.Timestamp("2019-01-01")
    assert first["origin"].min() == pd.Timestamp("2018-12-25")
    m = res.metrics
    assert set(m["test_year"]) == {"2018", "2019", "all"}
    assert set(res.dm_tests["reference"]) == {"persistence"}
    assert len(res.dm_tests) == 2 * 2  # two models x two horizons


def test_training_rows_end_before_the_test_year(series, settings, monkeypatch):
    fits = []
    real = bt.make_model

    def spy(name, feats, **kw):
        m = real(name, feats, **kw)
        fit = m.fit
        m.fit = lambda train, valid=None: (fits.append(train["target_date"].max()), fit(train, valid))[1]
        return m

    monkeypatch.setattr(bt, "make_model", spy)
    bt.run_backtest(series, ["ridge"], settings, [2019], do_tune=False)
    assert fits and all(d <= pd.Timestamp("2018-12-31") for d in fits)


def test_test_years_are_checked(series, settings):
    with pytest.raises(ValueError, match="not complete"):
        bt.run_backtest(series, ["persistence"], settings, [2020])
    with pytest.raises(ValueError, match="two earlier years"):
        bt.run_backtest(series, ["persistence"], settings, [2016])


def test_final_forecast_gives_one_row_per_horizon(series, settings):
    fc = final_forecast(series, "ridge", settings, do_tune=True)
    assert fc["horizon"].tolist() == [1, 7]
    assert fc["target_date"].tolist() == [pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-07").date()]
    assert (fc["forecast"] >= 0).all() and all("alpha" in p for p in fc["params"])


def test_lstm_optional(table):
    pytest.importorskip("torch")
    train, valid = table.iloc[:-200], table.iloc[-200:]
    a = make_model("lstm", FEATS, seed=0, hidden=8, epochs=3).fit(train, valid)
    b = make_model("lstm", FEATS, seed=0, hidden=8, epochs=3).fit(train, valid)
    pa, pb = a.predict(valid), b.predict(valid)
    assert pa.shape == (200,) and (pa >= 0).all()
    assert np.allclose(pa, pb)
    assert 1 <= a.best_epoch_ <= 3


def test_unknown_model(table):
    with pytest.raises(ValueError):
        make_model("transformer", FEATS)
