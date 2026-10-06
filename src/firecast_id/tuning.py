"""Hyperparameter search on a validation year.

The search fits each candidate on rows whose target date is before the validation year and
scores it on the validation year. The test year is never read here. The result, with every
trial, goes into the backtest report so that a run can be repeated.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae
from .models import make_model, search_space


@dataclass
class TuneResult:
    model: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_valid_mae: float | None = None
    trials: list[dict[str, Any]] = field(default_factory=list)


def candidates(space: dict[str, list[Any]], n_trials: int, seed: int) -> list[dict[str, Any]]:
    if not space:
        return []
    keys = sorted(space)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    if len(grid) <= n_trials:
        return grid
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(grid), size=n_trials, replace=False)
    return [grid[i] for i in sorted(pick)]


def tune(name: str, features: list[str], train: pd.DataFrame, valid: pd.DataFrame,
         n_trials: int, seed: int = 42, threads: int = 1) -> TuneResult:
    res = TuneResult(model=name)
    cands = candidates(search_space(name), n_trials, seed)
    if not cands:
        return res
    if valid.empty or train.empty:
        raise ValueError("tuning needs training rows and validation rows")
    best = np.inf
    for params in cands:
        m = make_model(name, features, seed=seed, threads=threads, **params).fit(train, valid)
        score = mae(valid["y"], m.predict(valid))
        trial = {"params": params, "valid_mae": score}
        if hasattr(m, "best_epoch_"):
            trial["params"] = {**params, "epochs": int(m.best_epoch_)}
        res.trials.append(trial)
        if score < best:
            best, res.best_params, res.best_valid_mae = score, trial["params"], score
    return res
