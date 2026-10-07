"""LSTM for direct multi-horizon forecasts (extra ``lstm``). torch is imported inside ``fit``.

The network reads the input window (imputed with train medians, scaled with train statistics, plus
a missing-value flag for each feature) and outputs all H temperature changes at once. Early
stopping uses the REAL validation samples and keeps the best weights.
"""

from __future__ import annotations

import numpy as np

from .models import Model
from .windows import Samples


class LSTMDirect(Model):
    name = "lstm"

    def __init__(self, hidden: int = 64, epochs: int = 30, lr: float = 2e-3, batch: int = 256, patience: int = 4,
                 seed: int = 0, name: str = "lstm"):
        self.hidden, self.epochs, self.lr, self.batch, self.patience, self.seed = hidden, epochs, lr, batch, patience, seed
        self.name = name

    def _prep(self, s: Samples) -> np.ndarray:
        X = s.X_seq
        miss = np.isnan(X).astype(np.float32)
        X = np.where(np.isnan(X), self.median, X)
        X = (X - self.mean) / self.std
        cal = np.repeat(s.calendar[:, None, :], X.shape[1], axis=1)
        return np.concatenate([X, miss, cal], axis=2).astype(np.float32)

    def fit(self, train: Samples, val: Samples | None = None) -> "LSTMDirect":
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('the LSTM needs: pip install -e ".[lstm]"') from exc
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        flat = train.X_seq.reshape(-1, train.X_seq.shape[2])
        self.median = np.nan_to_num(np.nanmedian(flat, axis=0))
        filled = np.where(np.isnan(flat), self.median, flat)
        self.mean, self.std = filled.mean(axis=0), filled.std(axis=0) + 1e-6
        Xtr = torch.tensor(self._prep(train))
        Rtr = torch.tensor((train.Y - train.y0[:, None]).astype(np.float32))
        H = train.Y.shape[1]

        class Net(nn.Module):
            def __init__(self, n_in, hidden):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                self.head = nn.Linear(hidden, H)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = Net(Xtr.shape[2], self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        has_val = val is not None and len(val) > 0
        if has_val:
            Xv = torch.tensor(self._prep(val))
            Rv = torch.tensor((val.Y - val.y0[:, None]).astype(np.float32))
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for _ in range(self.epochs):
            net.train()
            order = rng.permutation(len(Xtr))
            for k in range(0, len(order), self.batch):
                idx = torch.tensor(order[k : k + self.batch])
                opt.zero_grad()
                loss = loss_fn(net(Xtr[idx]), Rtr[idx])
                loss.backward()
                opt.step()
            if has_val:
                net.eval()
                with torch.no_grad():
                    v = float(loss_fn(net(Xv), Rv))
                self.history.append(v)
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def predict(self, s: Samples) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.net(torch.tensor(self._prep(s))).numpy()
        return out.astype(np.float64) + s.y0[:, None]
