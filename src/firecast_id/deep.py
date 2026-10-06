"""LSTM forecaster (PyTorch, optional extra ``deep``). This module imports torch at import time.

The LSTM reads the lag window as a sequence. The other features (rolling means, FRP, calendar,
climate) join the last hidden state. The output is a log rate, trained with a Poisson
likelihood. Scaling and imputation statistics come from the training rows only. If validation
rows are given, training stops when the validation loss does not improve for ``patience`` epochs.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import Forecaster


class _Net(nn.Module):
    def __init__(self, n_static: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


class LSTMForecaster(Forecaster):
    name = "lstm"
    space = {"hidden": [16, 32], "lr": [1e-3, 3e-3]}

    def _split(self, rows: pd.DataFrame):
        lag_cols = sorted([c for c in self.features if c.startswith("lag_")], key=lambda c: -int(c[4:]))
        static_cols = [c for c in self.features if not c.startswith("lag_")]
        return lag_cols, static_cols

    def _tensors(self, rows: pd.DataFrame):
        lag_cols, static_cols = self._split(rows)
        seq = rows[lag_cols].to_numpy(dtype=np.float32)
        seq = (seq - self.seq_mean_) / self.seq_std_
        st = rows[static_cols].to_numpy(dtype=np.float32)
        st = np.where(np.isnan(st), self.st_median_, st)
        st = (st - self.st_mean_) / self.st_std_
        return torch.from_numpy(seq[:, :, None]), torch.from_numpy(st.astype(np.float32))

    def fit(self, train, valid=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self.threads)
        lag_cols, static_cols = self._split(train)
        seq = train[lag_cols].to_numpy(dtype=np.float32)
        self.seq_mean_, self.seq_std_ = float(seq.mean()), float(seq.std() or 1.0)
        st = train[static_cols].to_numpy(dtype=np.float32)
        self.st_median_ = np.nan_to_num(np.nanmedian(st, axis=0)).astype(np.float32)
        st = np.where(np.isnan(st), self.st_median_, st)
        self.st_mean_ = st.mean(axis=0)
        self.st_std_ = np.where(st.std(axis=0) == 0, 1.0, st.std(axis=0)).astype(np.float32)

        hidden = int(self.params.get("hidden", 32))
        lr = float(self.params.get("lr", 3e-3))
        epochs = int(self.params.get("epochs", 40))
        patience = int(self.params.get("patience", 5))
        batch = int(self.params.get("batch", 256))
        self.net_ = _Net(len(static_cols), hidden)
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr)
        loss_fn = nn.PoissonNLLLoss(log_input=True)
        Xs, Xt = self._tensors(train)
        y = torch.tensor(train["y"].to_numpy(dtype=np.float32))
        if valid is not None and len(valid):
            Vs, Vt = self._tensors(valid)
            vy = torch.tensor(valid["y"].to_numpy(dtype=np.float32))
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        self.epochs_run_ = 0
        self.best_epoch_ = epochs
        for _ in range(epochs):
            self.net_.train()
            perm = torch.randperm(len(y), generator=gen)
            for i in range(0, len(y), batch):
                b = perm[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xs[b], Xt[b]), y[b])
                loss.backward()
                opt.step()
            self.epochs_run_ += 1
            if valid is not None and len(valid):
                self.net_.eval()
                with torch.no_grad():
                    vl = float(loss_fn(self.net_(Vs, Vt), vy))
                if vl < best - 1e-4:
                    best, best_state, wait = vl, copy.deepcopy(self.net_.state_dict()), 0
                    self.best_epoch_ = self.epochs_run_
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, rows):
        self.net_.eval()
        s, t = self._tensors(rows)
        with torch.no_grad():
            return torch.exp(self.net_(s, t)).numpy().astype(float)
