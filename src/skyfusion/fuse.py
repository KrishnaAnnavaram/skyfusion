"""Align the two sources on one UTC hourly grid, apply quality control and fuse like-for-like only.

Fusion rules:
- The forecast TARGET is the observed station temperature ``st_temp_c``. It is never averaged with
  the reanalysis temperature, so single-source and fused experiments predict the same quantity.
- Two columns are averaged only if they measure the same physical quantity at the same height:
  relative humidity at 2 m, precipitation in mm/h and SURFACE pressure.
- Sea-level pressure (station) and surface pressure (POWER) are different quantities. They stay in
  separate columns. Wind at 10 m (station) and wind at 2 m (POWER) also stay separate.
- Feature gaps are filled only FORWARD (up to 3 hours), because a backward fill or an interpolation
  uses future values. The target is never filled. A ``*_missing`` mask column records each gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POWER_RENAME = {
    "T2M": "pw_t2m_c", "RH2M": "pw_rh_pct", "WS2M": "pw_wind2_ms", "PS": "pw_sfc_kpa",
    "PRECTOTCORR": "pw_rain_mm", "ALLSKY_SFC_SW_DWN": "pw_sw_wm2", "ALLSKY_SFC_LW_DWN": "pw_lw_wm2",
}

# fused column -> (station column, POWER column); both columns hold the same physical quantity
FUSION_RULES: dict[str, tuple[str, str]] = {
    "fz_rh_pct": ("st_rh_pct", "pw_rh_pct"),
    "fz_rain_mm": ("st_rain_mm", "pw_rain_mm"),
    "fz_sfc_kpa": ("st_sfc_kpa", "pw_sfc_kpa"),
}

# physical limits for quality control (inclusive)
LIMITS: dict[str, tuple[float, float]] = {
    "temp_c": (-60, 60), "t2m_c": (-60, 60), "dew_point_c": (-80, 40), "feels_like_c": (-80, 75),
    "rh_pct": (0, 100), "slp_kpa": (85, 110), "sfc_kpa": (50, 110), "wind10_ms": (0, 80), "wind2_ms": (0, 80),
    "rain_mm": (0, 300), "sw_wm2": (0, 1500), "lw_wm2": (0, 800), "clouds_pct": (0, 100),
}

TARGET = "st_temp_c"
STATION_DROP = ("st_temp_min_c", "st_temp_max_c")  # aggregates over the hour, not observations


@dataclass
class QCReport:
    hours: int
    start: str
    end: str
    out_of_range: dict[str, int]
    missing_share: dict[str, float]
    target_missing: int


def _qc(df: pd.DataFrame) -> dict[str, int]:
    bad = {}
    for col in df.columns:
        key = col.split("_", 1)[1] if col[:3] in ("st_", "pw_") else col
        if key in LIMITS:
            lo, hi = LIMITS[key]
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                bad[col] = int(mask.sum())
                df.loc[mask, col] = np.nan
    return bad


def build_dataset(station: pd.DataFrame, power: pd.DataFrame | None, ffill_limit: int = 3) -> tuple[pd.DataFrame, QCReport]:
    """Return the hourly UTC table with ``st_*``, ``pw_*``, ``fz_*`` columns, masks and the target."""
    for name, df in (("station", station), ("power", power)):
        if df is not None and (df.index.tz is None or str(df.index.tz) != "UTC"):
            raise ValueError(f"the {name} index must be timezone-aware UTC")
    st = station.add_prefix("st_").drop(columns=list(STATION_DROP), errors="ignore")
    start, end = st.index.min(), st.index.max()
    if power is not None:
        pw = power.rename(columns=POWER_RENAME)
        pw = pw[[c for c in POWER_RENAME.values() if c in pw.columns]]
        start, end = max(start, pw.index.min()), min(end, pw.index.max())
    grid = pd.date_range(start.ceil("h"), end.floor("h"), freq="h", tz="UTC", name="time_utc")
    st = st[~st.index.duplicated()].reindex(grid)
    out = st
    if power is not None:
        out = out.join(pw[~pw.index.duplicated()].reindex(grid))
    bad = _qc(out)
    target_missing = int(out[TARGET].isna().sum())
    for col, (a, b) in FUSION_RULES.items():
        if a in out and b in out:
            out[col] = out[[a, b]].mean(axis=1, skipna=True)
    missing_share = {c: float(out[c].isna().mean()) for c in out.columns}
    features = [c for c in out.columns if c != TARGET]
    for c in features:
        if out[c].isna().any():
            out[f"{c}_missing"] = out[c].isna().astype(float)
            out[c] = out[c].ffill(limit=ffill_limit)
    out["target_temp_c"] = out[TARGET]
    report = QCReport(len(out), str(grid[0]), str(grid[-1]), bad, missing_share, target_missing)
    return out, report


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Station-only and fused feature lists. Both include the station temperature history."""
    cols = [c for c in df.columns if c != "target_temp_c"]
    station = [c for c in cols if c.startswith("st_")]
    fused = station + [c for c in cols if c.startswith(("pw_", "fz_"))]
    return {"station": station, "fused": fused}
