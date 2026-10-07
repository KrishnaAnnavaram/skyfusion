"""Forecast models with one interface: ``fit(train, val)`` and ``predict(samples) -> (n, H)``.

Baselines come first: persistence, seasonal naive and hour-of-day climatology. The learned models
(ridge, gradient boosting, LSTM) predict the CHANGE from the last observed temperature, and they fit
all preprocessing (imputation, scaling) on the train samples only. The validation samples select
the hyperparameters; they are never used to fit the final weights of ridge and boosting.
"""

from __future__ import annotations

from itertools import islice

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .windows import Samples


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


class Model:
    name = "base"

    def fit(self, train: Samples, val: Samples | None = None) -> "Model":
        return self

    def predict(self, s: Samples) -> np.ndarray:
        raise NotImplementedError


class Persistence(Model):
    name = "persistence"

    def predict(self, s):
        return np.repeat(s.y0[:, None], s.Y.shape[1], axis=1)


class SeasonalNaive(Model):
    """y(t+h) = y(t+h-24): the same hour yesterday. Needs a window of at least 24 hours."""

    name = "seasonal-naive"

    def predict(self, s):
        W, H = s.y_hist.shape[1], s.Y.shape[1]
        if W < 24 or H > 24:
            raise ValueError("seasonal naive needs window >= 24 and horizon <= 24")
        pred = s.y_hist[:, W - 24 : W - 24 + H].copy()
        # A gap in the history falls back to persistence.
        bad = np.isnan(pred)
        pred[bad] = np.repeat(s.y0[:, None], H, axis=1)[bad]
        return pred


class Climatology(Model):
    """Mean train temperature for each (month, UTC hour) of the target time."""

    name = "climatology"

    def fit(self, train, val=None):
        H = train.Y.shape[1]
        times = train.origins
        self.table = np.full((13, 24), np.nan)
        sums, counts = np.zeros((13, 24)), np.zeros((13, 24))
        for h in range(1, H + 1):
            tt = times + np.timedelta64(h, "h")
            np.add.at(sums, (tt.month.to_numpy(), tt.hour.to_numpy()), train.Y[:, h - 1])
            np.add.at(counts, (tt.month.to_numpy(), tt.hour.to_numpy()), 1)
        self.table = np.where(counts > 0, sums / np.maximum(counts, 1), np.nanmean(train.Y))
        return self

    def predict(self, s):
        H = s.Y.shape[1]
        out = np.empty((len(s), H))
        for h in range(1, H + 1):
            tt = s.origins + np.timedelta64(h, "h")
            out[:, h - 1] = self.table[tt.month.to_numpy(), tt.hour.to_numpy()]
        return out


class RidgeDirect(Model):
    """Imputer + scaler + ridge on the flattened window. Alpha is chosen on the validation set."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0), last_hours: int | None = None, name: str = "ridge"):
        self.alphas, self.last_hours, self.name = alphas, last_hours, name

    def _pipe(self, alpha):
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, train, val=None):
        Xtr, Rtr = train.flat(self.last_hours), train.Y - train.y0[:, None]
        best = None
        for a in self.alphas:
            p = self._pipe(a).fit(Xtr, Rtr)
            if val is None or len(val) == 0:
                best = (0.0, a, p)
                break
            score = _rmse(p.predict(val.flat(self.last_hours)) + val.y0[:, None], val.Y)
            if best is None or score < best[0]:
                best = (score, a, p)
        self.alpha, self.pipe = best[1], best[2]
        return self

    def predict(self, s):
        return self.pipe.predict(s.flat(self.last_hours)) + s.y0[:, None]


class GBMDirect(Model):
    """One histogram gradient boosting model for each horizon. NaN inputs are handled natively.

    It uses the last ``last_hours`` hours of the window to keep training fast. The number of boosting
    iterations for each horizon is the one with the lowest validation error (``staged_predict``).
    """

    def __init__(self, last_hours: int = 6, max_iter: int = 150, learning_rate: float = 0.1, seed: int = 0,
                 name: str = "gbm"):
        self.last_hours, self.max_iter, self.lr, self.seed, self.name = last_hours, max_iter, learning_rate, seed, name

    def fit(self, train, val=None):
        # Many OpenMP threads make small boosting jobs slower on hosts with many cores.
        with threadpool_limits(4):
            return self._fit(train, val)

    def _fit(self, train, val=None):
        Xtr, Rtr = train.flat(self.last_hours), train.Y - train.y0[:, None]
        self.models, self.best_iter = [], []
        Xv = val.flat(self.last_hours) if val is not None and len(val) else None
        for h in range(train.Y.shape[1]):
            m = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.lr, random_state=self.seed,
                                              early_stopping=False).fit(Xtr, Rtr[:, h])
            best = m.n_iter_
            if Xv is not None:
                errs = [np.mean((p + val.y0 - val.Y[:, h]) ** 2) for p in m.staged_predict(Xv)]
                best = int(np.argmin(errs)) + 1
            self.models.append(m)
            self.best_iter.append(best)
        return self

    def predict(self, s):
        X = s.flat(self.last_hours)
        cols = []
        for m, best in zip(self.models, self.best_iter):
            cols.append(next(islice(m.staged_predict(X), best - 1, None)))
        return np.column_stack(cols) + s.y0[:, None]
