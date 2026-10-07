import numpy as np
import pandas as pd
import pytest

from skyfusion.config import Settings
from skyfusion.fuse import FUSION_RULES, build_dataset, feature_sets
from skyfusion.ingest import IngestError, fetch_power, parse_power_text, power_url, read_openweather_csv

POWER_LST = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Hourly Data
Dates (month/day/year): 01/01/2021 through 01/01/2021 in LST
-END HEADER-
YEAR,MO,DY,HR,T2M,PS,ALLSKY_SFC_SW_DWN
2021,1,1,0,2.57,97.98,-999
2021,1,1,1,2.15,98.05,0.0
"""


def test_power_lst_is_shifted_to_utc():
    # Problem 1: LST 00:00 at lon -97 is 06:00 UTC.
    df = parse_power_text(POWER_LST, lon=-97.04)
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2021-01-01 06:00", tz="UTC")
    assert df.T2M.iloc[0] == 2.57


def test_power_fill_value_becomes_nan():
    # Problem 8: -999 is a fill value, not a measurement.
    df = parse_power_text(POWER_LST, lon=-97.04)
    assert np.isnan(df.ALLSKY_SFC_SW_DWN.iloc[0]) and df.ALLSKY_SFC_SW_DWN.iloc[1] == 0.0


def test_power_utc_and_missing_header():
    utc = parse_power_text(POWER_LST.replace("in LST", "in UTC"), lon=-97.04)
    assert utc.index[0] == pd.Timestamp("2021-01-01 00:00", tz="UTC")
    with pytest.raises(IngestError, match="LST"):
        parse_power_text("YEAR,MO,DY,HR,T2M\n2021,1,1,0,1\n", lon=0)
    with pytest.raises(IngestError, match="misses columns"):
        parse_power_text("-BEGIN HEADER-\nin UTC\n-END HEADER-\nA,B\n1,2\n", lon=0)


def test_fetch_power_uses_utc_and_cache(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        return POWER_LST.replace("in LST", "in UTC")

    df = fetch_power(32.9, -97.04, "20210101", "20210101", tmp_path, transport)
    fetch_power(32.9, -97.04, "20210101", "20210101", tmp_path, transport)
    assert len(calls) == 1 and "time-standard=UTC" in calls[0] and len(df) == 2
    assert "parameters=T2M" in power_url(1, 2, "20200101", "20200102")
    with pytest.raises(IngestError, match="API error"):
        fetch_power(1, 2, "x", "y", tmp_path, lambda url: '{"messages": ["bad date"]}')


def test_openweather_units_and_pressures(tables):
    truth, st, _ = tables
    common = st.index.intersection(truth.index)
    assert np.abs(st.loc[common, "temp_c"] - truth.loc[common, "temp_c"]).mean() < 0.5  # kelvin -> C
    assert 100 < st.slp_kpa.median() < 104  # sea level, kPa
    assert "sfc_kpa" not in st  # this export has no grnd_level
    assert st.rain_mm.isna().sum() == 0


def test_openweather_dt_iso_fallback(tmp_path):
    p = tmp_path / "ow.csv"
    p.write_text("dt_iso,temp,pressure\n2021-01-01 00:00:00 +0000 UTC,273.15,1013\n", encoding="utf-8")
    df = read_openweather_csv(p)
    assert df.index[0] == pd.Timestamp("2021-01-01", tz="UTC") and df.temp_c.iloc[0] == 0 and df.slp_kpa.iloc[0] == 101.3
    p.write_text("time,temp\n1,2\n", encoding="utf-8")
    with pytest.raises(IngestError):
        read_openweather_csv(p)


def test_sources_align_at_lag_zero(fused):
    # Problem 1: after the UTC shift, the station and POWER temperatures agree best at lag 0.
    df, _ = fused
    a, b = df.st_temp_c, df.pw_t2m_c
    corr = {lag: a.corr(b.shift(lag)) for lag in range(-8, 9)}
    assert max(corr, key=corr.get) == 0
    assert corr[0] > corr[6] + 0.05 and corr[0] > corr[-6] + 0.05


def test_fusion_rules_join_only_like_quantities(fused):
    # Problem 7: sea-level pressure is never averaged with surface pressure.
    df, _ = fused
    for col, (a, b) in FUSION_RULES.items():
        assert "slp" not in a and "slp" not in b and "wind" not in col
    assert "fz_sfc_kpa" not in df  # the station gives no surface pressure, so nothing is fused
    assert df.pw_sfc_kpa.median() < 100.5 < df.st_slp_kpa.median()
    assert "st_temp_min_c" not in df and "st_temp_max_c" not in df
    assert np.allclose(df.target_temp_c.dropna(), df.st_temp_c.dropna())


def test_target_is_never_filled_and_features_only_forward(fused, tables):
    df, report = fused
    _, st, _ = tables
    assert report.target_missing == int(df.target_temp_c.isna().sum()) > 0
    gap = df.index[df.st_rh_pct_missing == 1]
    first = gap[0]
    prev = first - pd.Timedelta(hours=1)
    assert df.loc[first, "st_rh_pct"] == df.loc[prev, "st_rh_pct"]  # forward fill
    outage = df.st_rh_pct.isna()
    assert outage.sum() > 24  # a multi-day outage stays a gap after 3 filled hours


def test_qc_and_timezone_checks(tables):
    _, st, pw = tables
    bad = st.copy()
    bad.iloc[10, bad.columns.get_loc("rh_pct")] = 150
    _, rep = build_dataset(bad, None)
    assert rep.out_of_range == {"st_rh_pct": 1}
    naive = st.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        build_dataset(naive, pw)


def test_feature_sets(fused):
    sets = feature_sets(fused[0])
    assert set(sets["station"]) < set(sets["fused"])
    assert all(c.startswith("st_") for c in sets["station"])
    assert "target_temp_c" not in sets["fused"]


def test_settings(monkeypatch):
    assert Settings().lst_offset_hours == -6
    monkeypatch.setenv("SKYFUSION_TRAIN_END", "2022")
    monkeypatch.setenv("SKYFUSION_VAL_END", "2021")
    with pytest.raises(ValueError):
        Settings.from_env()
