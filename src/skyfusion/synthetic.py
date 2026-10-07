"""Synthetic station and NASA POWER files in their REAL raw formats, from one hidden truth.

The truth is an hourly UTC series with seasonal and daily cycles, a cloud process, weather fronts and
humidity. The station file looks like an OpenWeather History Bulk export (UTC ``dt``, kelvin, hPa,
sea-level ``pressure``). The POWER file looks like a POWER point download in LOCAL STANDARD TIME with
a header and -999 fill values. Thus the tests run the same loaders as real data.

Clouds lower the daytime heating some hours later, and only POWER sees clouds (shortwave
irradiance). So POWER can add skill for later horizons. The size of that skill is a property of this
generator, not of real weather.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ELEVATION_M = 183.5


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(1 - phi**2)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + e[k]
    return out


def make_truth(start: str = "2018-01-01", end: str = "2022-12-31 23:00", lon: float = -97.04, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(t)
    local_hour = (t.hour.to_numpy() + round(lon / 15)) % 24
    doy = t.dayofyear.to_numpy()
    cloud = 1 / (1 + np.exp(-_ar1(n, 0.97, 0.45, rng)))  # 0..1, persistent
    lagged_cloud = np.concatenate([np.full(3, cloud[0]), cloud[:-3]])  # clouds act about 3 h later
    seasonal = 19 - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = (6.5 * (1 - 0.7 * lagged_cloud)) * np.cos(2 * np.pi * (local_hour - 15) / 24)
    front = _ar1(n, 0.995, 0.25, rng)
    temp = seasonal + diurnal + front
    rh = np.clip(62 - 2.2 * (temp - seasonal) + 15 * cloud + _ar1(n, 0.9, 2.0, rng), 8, 100)
    a, b = 17.62, 243.12
    gamma = np.log(rh / 100) + a * temp / (b + temp)
    dew = b * gamma / (a - gamma)
    sfc = 99.3 + _ar1(n, 0.99, 0.05, rng) - 0.02 * front
    slp = sfc + ELEVATION_M / 8.3 / 10  # about +2.2 kPa at 183 m
    wind10 = np.abs(3.5 + 1.5 * cloud + _ar1(n, 0.9, 0.6, rng))
    rain = np.where((cloud > 0.85) & (rng.random(n) < 0.3), rng.gamma(1.2, 1.5, n), 0.0)
    elev = np.clip(np.cos(2 * np.pi * (local_hour - 12.5) / 24), 0, None) * (0.55 + 0.45 * np.cos(2 * np.pi * (doy - 172) / 365.25))
    sw_clear = 1000 * elev
    return pd.DataFrame({
        "temp_c": temp, "rh_pct": rh, "dew_c": dew, "sfc_kpa": sfc, "slp_kpa": slp, "wind10_ms": wind10,
        "rain_mm": rain, "cloud": cloud, "sw_wm2": sw_clear * (1 - 0.75 * cloud),
        "lw_wm2": 300 + 2.5 * temp + 60 * cloud,
    }, index=pd.DatetimeIndex(t, name="time_utc"))


def write_openweather(truth: pd.DataFrame, path: str | Path, seed: int = 7, outage_days: int = 5) -> Path:
    """Station file: UTC, kelvin, hPa sea-level pressure, 1 % random gaps and one multi-day outage."""
    rng = np.random.default_rng(seed + 1)
    n = len(truth)
    keep = rng.random(n) > 0.01
    start = int(rng.integers(n // 3, n // 2))
    keep[start : start + 24 * outage_days] = False
    tr = truth[keep]
    temp_k = tr.temp_c + rng.normal(0, 0.3, len(tr)) + 273.15
    df = pd.DataFrame({
        "dt": ((tr.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(np.int64),
        "dt_iso": tr.index.strftime("%Y-%m-%d %H:%M:%S +0000 UTC"),
        "timezone": -18000,
        "city_name": "Synthetic Site",
        "temp": temp_k.round(2),
        "dew_point": (tr.dew_c + 273.15 + rng.normal(0, 0.4, len(tr))).round(2),
        "feels_like": (temp_k - 1.5 * (tr.wind10_ms > 4)).round(2),
        "temp_min": (temp_k - 0.5).round(2),
        "temp_max": (temp_k + 0.5).round(2),
        "pressure": (tr.slp_kpa * 10).round(0),
        "humidity": np.clip(tr.rh_pct + rng.normal(0, 2, len(tr)), 0, 100).round(0),
        "wind_speed": (tr.wind10_ms + rng.normal(0, 0.3, len(tr))).clip(lower=0).round(2),
        "rain_1h": np.where(tr.rain_mm > 0, tr.rain_mm.round(2), np.nan),
    })
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_power(truth: pd.DataFrame, path: str | Path, lon: float = -97.04, seed: int = 7, fill_last_days: int = 60) -> Path:
    """POWER file in local standard time with a header block and -999 fill values."""
    rng = np.random.default_rng(seed + 2)
    smooth = truth.rolling(3, center=True, min_periods=1).mean()
    n = len(truth)
    sw = (smooth.sw_wm2 + rng.normal(0, 15, n)).clip(lower=0)
    lw = smooth.lw_wm2 + rng.normal(0, 5, n)
    sw.iloc[-24 * fill_last_days:] = -999.0
    lw.iloc[-24 * fill_last_days:] = -999.0
    local = truth.index.tz_convert(None) + pd.Timedelta(hours=round(lon / 15))
    df = pd.DataFrame({
        "YEAR": local.year, "MO": local.month, "DY": local.day, "HR": local.hour,
        "ALLSKY_SFC_LW_DWN": lw.round(2).to_numpy(),
        "ALLSKY_SFC_SW_DWN": sw.round(2).to_numpy(),
        "T2M": (smooth.temp_c + 0.6 + rng.normal(0, 0.5, n)).round(2).to_numpy(),
        "WS2M": (smooth.wind10_ms * 0.75 + rng.normal(0, 0.2, n)).clip(lower=0).round(2).to_numpy(),
        "RH2M": np.clip(smooth.rh_pct + rng.normal(0, 3, n), 0, 100).round(2).to_numpy(),
        "PS": (smooth.sfc_kpa + rng.normal(0, 0.03, n)).round(2).to_numpy(),
        "PRECTOTCORR": (smooth.rain_mm * 0.8).round(2).to_numpy(),
    })
    first, last = local[0], local[-1]
    header = (
        "-BEGIN HEADER-\nNASA/POWER Source Native Resolution Hourly Data (SYNTHETIC COPY FOR TESTS)\n"
        f"Dates (month/day/year): {first:%m/%d/%Y} through {last:%m/%d/%Y} in LST\n"
        "Location: Latitude  32.8998   Longitude -97.0403\n"
        "The value for missing source data that cannot be computed or is outside of the sources availability range: -999\n"
        "-END HEADER-\n"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header + df.to_csv(index=False), encoding="utf-8")
    return p


def write_synthetic(folder: str | Path, start: str = "2018-01-01", end: str = "2022-12-31 23:00", seed: int = 7) -> tuple[Path, Path]:
    truth = make_truth(start, end, seed=seed)
    folder = Path(folder)
    return write_openweather(truth, folder / "station_openweather.csv", seed), write_power(truth, folder / "power_lst.csv", seed=seed)
