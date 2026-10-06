from dataclasses import replace

import pytest

from firecast_id import synthetic
from firecast_id.config import Settings


@pytest.fixture(scope="session")
def synth():
    return synthetic.generate("2015-01-01", "2019-12-31", seed=3)


@pytest.fixture(scope="session")
def series(synth):
    return synth[0]


@pytest.fixture(scope="session")
def settings():
    return replace(Settings(), horizons=(1, 7), n_lags=7, tuning_trials=2)
