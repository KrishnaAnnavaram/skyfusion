"""Per-horizon scores, skill against persistence and the single-source vs fused ablation.

Each model is scored on ITS OWN predictions, on the same test samples and the same target. The
ablation uses identical samples (same origins, same target) and changes only the input features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fuse import feature_sets
from .models import Climatology, GBMDirect, Model, Persistence, RidgeDirect, SeasonalNaive
from .windows import Samples, SplitSamples, make_samples, split_by_year

REPORT_HORIZONS = (1, 3, 6, 12, 24)


@dataclass
class Scores:
    model: str
    features: str
    mae: np.ndarray  # (H,)
    rmse: np.ndarray  # (H,)
    skill: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 1 - RMSE / RMSE(persistence)

    def at(self, h: int) -> tuple[float, float, float]:
        return float(self.mae[h - 1]), float(self.rmse[h - 1]), float(self.skill[h - 1]) if self.skill.size else float("nan")

    def as_dict(self) -> dict:
        return {"model": self.model, "features": self.features, "mae": self.mae.round(4).tolist(),
                "rmse": self.rmse.round(4).tolist(), "skill": self.skill.round(4).tolist()}


def score(pred: np.ndarray, s: Samples, model: str, features: str) -> Scores:
    if pred.shape != s.Y.shape:
        raise ValueError(f"prediction shape {pred.shape} differs from target shape {s.Y.shape}")
    err = pred - s.Y
    return Scores(model, features, np.abs(err).mean(axis=0), np.sqrt((err**2).mean(axis=0)))


def add_skill(rows: list[Scores]) -> list[Scores]:
    ref = next(r for r in rows if r.model == "persistence")
    for r in rows:
        r.skill = 1 - r.rmse / ref.rmse
    return rows


def block_bootstrap_diff(pred_a: np.ndarray, pred_b: np.ndarray, s: Samples, horizon: int, n_boot: int = 500,
                         seed: int = 0) -> tuple[float, float, float]:
    """RMSE(a) - RMSE(b) at one horizon, with a 95 % interval from a bootstrap over whole days."""
    days = s.origins.normalize()
    codes, uniq = pd.factorize(days)
    ea = (pred_a[:, horizon - 1] - s.Y[:, horizon - 1]) ** 2
    eb = (pred_b[:, horizon - 1] - s.Y[:, horizon - 1]) ** 2
    sa = np.bincount(codes, ea, len(uniq))
    sb = np.bincount(codes, eb, len(uniq))
    cnt = np.bincount(codes, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        n = cnt[pick].sum()
        diffs.append(np.sqrt(sa[pick].sum() / n) - np.sqrt(sb[pick].sum() / n))
    point = np.sqrt(ea.mean()) - np.sqrt(eb.mean())
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def default_models(seed: int = 0, include_lstm: bool = False, fast: bool = False) -> list[Model]:
    models: list[Model] = [Persistence(), SeasonalNaive(), Climatology(), RidgeDirect()]
    models.append(GBMDirect(seed=seed, max_iter=60 if fast else 150))
    if include_lstm:
        from .lstm import LSTMDirect

        models.append(LSTMDirect(seed=seed, epochs=5 if fast else 30))
    return models


LEARNED = ("ridge", "gbm", "lstm")


@dataclass
class Experiment:
    splits: dict[str, SplitSamples]  # feature set -> split samples
    rows: list[Scores]
    ablation: list[dict]


def run_experiment(df: pd.DataFrame, window: int, horizon: int, train_end: int, val_end: int, seed: int = 0,
                   include_lstm: bool = False, fast: bool = False) -> Experiment:
    sets = feature_sets(df)
    splits = {name: split_by_year(make_samples(df, cols, window, horizon), train_end, val_end, horizon)
              for name, cols in sets.items()}
    # identical samples in both feature sets
    if not splits["station"].test.origins.equals(splits["fused"].test.origins):
        raise AssertionError("the feature sets must share the same test origins")
    rows: list[Scores] = []
    preds: dict[tuple[str, str], np.ndarray] = {}
    for fname in ("station", "fused"):
        sp = splits[fname]
        for model in default_models(seed, include_lstm, fast):
            if fname == "fused" and model.name not in LEARNED:
                continue  # baselines do not use the extra features
            model.fit(sp.train, sp.val)
            pred = model.predict(sp.test)
            preds[(model.name, fname)] = pred
            rows.append(score(pred, sp.test, model.name, fname))
    add_skill(rows)
    ablation = []
    test = splits["station"].test
    for name in LEARNED:
        if (name, "station") in preds:
            for h in (1, 6, 24) if horizon >= 24 else (1, horizon):
                d, lo, hi = block_bootstrap_diff(preds[(name, "fused")], preds[(name, "station")], test, h, seed=seed)
                ablation.append({"model": name, "horizon": h, "rmse_fused_minus_station": d, "ci95": [lo, hi]})
    return Experiment(splits, rows, ablation)
