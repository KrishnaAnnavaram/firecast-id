"""Problem 6 (invalid significance tests): a Diebold-Mariano test on daily errors."""
import numpy as np
import pytest

from firecast_id.metrics import diebold_mariano, mae, mase, mase_scale, poisson_deviance, rmse


def test_basic_metrics():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert mase_scale([1, 3, 2]) == pytest.approx(1.5)
    assert mase(y, p, 2.0) == pytest.approx(0.5)
    assert np.isnan(mase_scale([5, 5, 5]))


def test_poisson_deviance_is_zero_for_a_perfect_forecast():
    y = np.array([0.0, 3.0, 10.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)
    assert poisson_deviance(y, y + 1) > 0


def test_dm_identical_errors_give_no_difference():
    e = np.random.default_rng(0).normal(size=100)
    r = diebold_mariano(e, e)
    assert r.p_value == 1.0 and r.mean_diff == 0.0


def test_dm_finds_a_clearly_better_model():
    rng = np.random.default_rng(1)
    small, large = rng.normal(0, 1, 400), rng.normal(0, 3, 400)
    r = diebold_mariano(small, large, horizon=1)
    assert r.statistic < 0 and r.p_value < 0.001 and r.mean_diff < 0


def test_dm_matches_a_hand_calculation_for_h1():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 50), rng.normal(0, 1.2, 50)
    d = np.abs(a) - np.abs(b)
    n = len(d)
    stat = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / n) * np.sqrt((n - 1) / n)
    assert diebold_mariano(a, b, horizon=1).statistic == pytest.approx(stat)


def test_dm_needs_aligned_inputs():
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(20), np.zeros(19))
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(20), np.zeros(20), loss="other")
