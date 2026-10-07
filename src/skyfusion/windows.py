"""Supervised samples for DIRECT multi-horizon forecasting, and the time-based split.

For a forecast origin t (the last observed hour), the inputs are the hours t-W+1 .. t and the targets
are the station temperatures at t+1 .. t+H. All H horizons come from one model call, so there is no
recursive loop that feeds predictions back as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TARGET_COL = "target_temp_c"


@dataclass
class Samples:
    origins: pd.DatetimeIndex  # time t of each sample (UTC)
    X_seq: np.ndarray  # (n, W, F) input window, may contain NaN
    calendar: np.ndarray  # (n, 4) hour and day-of-year sin/cos at the origin
    y_hist: np.ndarray  # (n, W) target history (station temperature)
    Y: np.ndarray  # (n, H) future targets
    features: list[str]

    def __len__(self) -> int:
        return len(self.origins)

    @property
    def y0(self) -> np.ndarray:
        return self.y_hist[:, -1]

    def flat(self, last_hours: int | None = None) -> np.ndarray:
        seq = self.X_seq if last_hours is None else self.X_seq[:, -last_hours:, :]
        return np.hstack([seq.reshape(len(self), -1), self.calendar])

    def subset(self, mask: np.ndarray) -> "Samples":
        return Samples(self.origins[mask], self.X_seq[mask], self.calendar[mask], self.y_hist[mask], self.Y[mask], self.features)


def make_samples(df: pd.DataFrame, features: list[str], window: int = 24, horizon: int = 24) -> Samples:
    """Build all samples whose target history end and all H targets are present."""
    if TARGET_COL not in df:
        raise KeyError(f"the table needs the column {TARGET_COL!r}")
    idx = df.index
    if not (idx.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("the table must be on a regular hourly grid")
    F = df[features].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    n = len(df)
    last = n - horizon  # an origin needs H future rows
    if last <= window - 1:
        raise ValueError("the series is shorter than window + horizon")
    pos = np.arange(window - 1, last)
    Xw = sliding_window_view(F, (window, F.shape[1]))[:, 0]  # (n-W+1, W, F); row k ends at k+W-1
    yw = sliding_window_view(y, window)  # (n-W+1, W)
    fut = sliding_window_view(y[1:], horizon)  # row k = y[k+1 .. k+H]
    X_seq = Xw[pos - window + 1]
    y_hist = yw[pos - window + 1]
    Y = fut[pos]
    ok = ~np.isnan(Y).any(axis=1) & ~np.isnan(y_hist[:, -1])
    origins = idx[pos]
    hours = origins.hour.to_numpy()
    doy = origins.dayofyear.to_numpy()
    cal = np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
                           np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    return Samples(origins[ok], X_seq[ok], cal[ok], y_hist[ok], Y[ok], list(features))


@dataclass
class SplitSamples:
    train: Samples
    val: Samples
    test: Samples


def split_by_year(s: Samples, train_end: int, val_end: int, horizon: int) -> SplitSamples:
    """Train: years <= train_end. Validation: up to val_end. Test: later years.

    An embargo removes each sample whose target hours cross into the next part, so no target hour
    is in two parts.
    """
    years = s.origins.year.to_numpy()
    last_target = s.origins + pd.Timedelta(hours=horizon)
    lt_years = last_target.year.to_numpy()
    train = (years <= train_end) & (lt_years <= train_end)
    val = (years > train_end) & (years <= val_end) & (lt_years <= val_end)
    test = years > val_end
    if not train.any() or not val.any() or not test.any():
        raise ValueError(f"empty split: train {train.sum()}, validation {val.sum()}, test {test.sum()} samples")
    return SplitSamples(s.subset(train), s.subset(val), s.subset(test))
