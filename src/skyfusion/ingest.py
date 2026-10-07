"""Loaders for the two sources. Both return an hourly table with a UTC ``DatetimeIndex``.

NASA POWER hourly point files can be in local standard time (LST, no daylight saving) or in UTC.
The header says which. An LST file is shifted to UTC with the offset ``round(lon / 15)`` hours.
The prototype joined LST rows with UTC rows, so it averaged readings that were 6 hours apart.

OpenWeather "History Bulk" files give ``dt`` (Unix seconds, UTC). Temperatures are kelvin and
pressures are hPa. ``pressure`` is SEA-LEVEL pressure. ``grnd_level`` is surface pressure.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

POWER_FILL = -999.0
POWER_PARAMETERS = ("T2M", "RH2M", "WS2M", "PS", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN")
POWER_API = "https://power.larc.nasa.gov/api/temporal/hourly/point"


class IngestError(ValueError):
    pass


def _split_header(text: str) -> tuple[str, str]:
    if "-END HEADER-" in text:
        head, body = text.split("-END HEADER-", 1)
        return head, body.lstrip("\r\n")
    return "", text


def power_time_standard(header: str) -> str:
    m = re.search(r"\bin\s+(LST|UTC)\b", header)
    if not m:
        raise IngestError("the POWER header does not say 'in LST' or 'in UTC'; pass time_standard explicitly")
    return m.group(1)


def read_power_csv(path: str | Path, lon: float, time_standard: str | None = None) -> pd.DataFrame:
    """Read a POWER hourly point CSV file. -999 becomes NaN. The index is UTC."""
    return parse_power_text(Path(path).read_text(encoding="utf-8"), lon, time_standard)


def parse_power_text(text: str, lon: float, time_standard: str | None = None) -> pd.DataFrame:
    header, body = _split_header(text)
    std = time_standard or power_time_standard(header)
    df = pd.read_csv(io.StringIO(body))
    need = {"YEAR", "MO", "DY", "HR"}
    if not need <= set(df.columns):
        raise IngestError(f"POWER file misses columns {sorted(need - set(df.columns))}")
    local = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR))
    offset = int(round(lon / 15)) if std == "LST" else 0
    utc = (local - pd.Timedelta(hours=offset)).dt.tz_localize("UTC")
    values = df.drop(columns=["YEAR", "MO", "DY", "HR"]).astype(float).replace(POWER_FILL, np.nan)
    values.index = pd.DatetimeIndex(utc, name="time_utc")
    values = values[~values.index.duplicated(keep="first")].sort_index()
    return values


def read_power_files(paths, lon: float) -> pd.DataFrame:
    parts = [read_power_csv(Path(p), lon) for p in paths]
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="first")]


Transport = Callable[[str], str]


def _http_get(url: str) -> str:  # pragma: no cover - network
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed public API
        return resp.read().decode("utf-8")


def power_url(lat: float, lon: float, start: str, end: str, parameters=POWER_PARAMETERS) -> str:
    """URL of a POWER hourly point request in UTC (``time-standard=UTC``), CSV format."""
    return (f"{POWER_API}?parameters={','.join(parameters)}&community=RE&longitude={lon:.4f}&latitude={lat:.4f}"
            f"&start={start}&end={end}&format=CSV&time-standard=UTC")


def fetch_power(lat: float, lon: float, start: str, end: str, cache_dir: str | Path = "cache",
                transport: Transport | None = None) -> pd.DataFrame:
    """Download (or read from the cache) one POWER request. Dates are YYYYMMDD."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = f"power_{lat:.4f}_{lon:.4f}_{start}_{end}_UTC.csv"
    path = cache / key
    if not path.is_file():
        text = (transport or _http_get)(power_url(lat, lon, start, end))
        if text.lstrip().startswith("{"):
            raise IngestError(f"POWER API error: {json.loads(text).get('messages', text[:200])}")
        path.write_text(text, encoding="utf-8")
    return read_power_csv(path, lon, time_standard="UTC")


OPENWEATHER_KELVIN = ("temp", "dew_point", "feels_like", "temp_min", "temp_max")


def read_openweather_csv(path: str | Path) -> pd.DataFrame:
    """Parse an OpenWeather History Bulk CSV into SI units with a UTC index.

    Output columns (when present): ``temp_c``, ``dew_point_c``, ``feels_like_c``, ``rh_pct``,
    ``slp_kpa`` (sea level), ``sfc_kpa`` (surface, from grnd_level), ``wind10_ms``, ``rain_mm``,
    ``clouds_pct``. ``feels_like`` is a different quantity and is never mixed with ``temp``.
    """
    df = pd.read_csv(path)
    if "dt" in df.columns:
        idx = pd.to_datetime(df["dt"], unit="s", utc=True)
    elif "dt_iso" in df.columns:
        iso = df["dt_iso"].astype(str).str.replace(" UTC", "", regex=False)
        idx = pd.to_datetime(iso, utc=True)
    else:
        raise IngestError("OpenWeather file needs a 'dt' or 'dt_iso' column")
    out = pd.DataFrame(index=pd.DatetimeIndex(idx, name="time_utc"))
    for col in OPENWEATHER_KELVIN:
        if col in df.columns:
            out[f"{col}_c"] = df[col].to_numpy(dtype=float) - 273.15
    mapping = {"humidity": "rh_pct", "wind_speed": "wind10_ms", "rain_1h": "rain_mm", "clouds_all": "clouds_pct"}
    for src, dst in mapping.items():
        if src in df.columns:
            out[dst] = df[src].to_numpy(dtype=float)
    if "pressure" in df.columns:
        out["slp_kpa"] = df["pressure"].to_numpy(dtype=float) / 10
    if "grnd_level" in df.columns:
        out["sfc_kpa"] = df["grnd_level"].to_numpy(dtype=float) / 10
    if "rain_mm" in out:
        out["rain_mm"] = out["rain_mm"].fillna(0.0)  # OpenWeather leaves rain empty when it did not rain
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out
