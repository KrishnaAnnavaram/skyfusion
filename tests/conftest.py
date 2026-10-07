import os

import pytest

from skyfusion.fuse import build_dataset
from skyfusion.ingest import read_openweather_csv, read_power_csv
from skyfusion.synthetic import make_truth, write_openweather, write_power

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SKYFUSION_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    truth = make_truth("2018-01-01", "2020-03-31 23:00", seed=3)
    return truth, write_openweather(truth, d / "st.csv", seed=3), write_power(truth, d / "pw.csv", seed=3)


@pytest.fixture(scope="session")
def tables(files):
    truth, st_path, pw_path = files
    return truth, read_openweather_csv(st_path), read_power_csv(pw_path, lon=-97.04)


@pytest.fixture(scope="session")
def fused(tables):
    _, st, pw = tables
    df, report = build_dataset(st, pw)
    return df, report
