"""The ``skyfusion`` command line."""

from __future__ import annotations

import os

# joblib (used by scikit-learn) prints a long warning on some Windows hosts when it counts CPU cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import Settings
from .evaluate import REPORT_HORIZONS, run_experiment
from .fuse import build_dataset, feature_sets
from .ingest import fetch_power, read_openweather_csv, read_power_files
from .models import RidgeDirect
from .synthetic import write_synthetic
from .windows import make_samples, split_by_year


def cmd_synth(args) -> int:
    st, pw = write_synthetic(args.out, args.start, args.end, args.seed)
    print(f"wrote {st} (station, UTC) and {pw} (POWER, LST with -999 fill values)")
    return 0


def cmd_fetch_power(args) -> int:  # pragma: no cover - network
    s = Settings.from_env()
    df = fetch_power(s.lat, s.lon, args.start, args.end, s.cache_dir)
    print(f"{len(df)} hourly rows from {df.index.min()} to {df.index.max()} (UTC), cached in {s.cache_dir}")
    return 0


def _build(args, s: Settings):
    station = read_openweather_csv(args.station)
    power = read_power_files(args.power, s.lon) if args.power else None
    return build_dataset(station, power)


def cmd_build(args) -> int:
    s = Settings.from_env()
    df, report = _build(args, s)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"wrote {out}: {len(df)} hours, {len(df.columns)} columns, target missing {report.target_missing} hours")
    if report.out_of_range:
        print("values out of range set to NaN:", report.out_of_range)
    return 0


def cmd_qc(args) -> int:
    s = Settings.from_env()
    _, report = _build(args, s)
    print(json.dumps(asdict(report), indent=2))
    return 0


def _load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def cmd_evaluate(args) -> int:
    s = Settings.from_env()
    df = _load_table(args.dataset)
    exp = run_experiment(df, args.window or s.window, args.horizon or s.horizon, args.train_end or s.train_end,
                         args.val_end or s.val_end, seed=s.seed, include_lstm=args.lstm, fast=args.fast)
    if args.json:
        print(json.dumps({"rows": [r.as_dict() for r in exp.rows], "ablation": exp.ablation}, indent=2))
        return 0
    sp = exp.splits["station"]
    H = sp.test.Y.shape[1]
    print(f"samples: train {len(sp.train)}, validation {len(sp.val)}, test {len(sp.test)} "
          f"(test origins {sp.test.origins.min():%Y-%m-%d} to {sp.test.origins.max():%Y-%m-%d})")
    hs = [h for h in REPORT_HORIZONS if h <= H]
    print(f"{'model':16s} {'features':8s} " + " ".join(f"{'MAE@' + str(h):>7s} {'skill@' + str(h):>8s}" for h in hs))
    for r in exp.rows:
        print(f"{r.model:16s} {r.features:8s} " + " ".join(f"{r.at(h)[0]:7.3f} {r.at(h)[2]:8.3f}" for h in hs))
    print("\nablation: RMSE(fused) - RMSE(station), 95% day-block bootstrap interval (negative = fusion helps)")
    for a in exp.ablation:
        print(f"  {a['model']:6s} h={a['horizon']:2d}: {a['rmse_fused_minus_station']:+.3f} "
              f"[{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]")
    return 0


def cmd_forecast(args) -> int:
    """Fit ridge on train + validation years and forecast the next H hours from the last window."""
    s = Settings.from_env()
    df = _load_table(args.dataset)
    cols = feature_sets(df)["fused" if args.fused else "station"]
    samples = make_samples(df, cols, s.window, s.horizon)
    sp = split_by_year(samples, s.train_end, s.val_end, s.horizon)
    model = RidgeDirect().fit(sp.train, sp.val)
    last = samples.subset(samples.origins == samples.origins.max())
    pred = model.predict(last)[0]
    origin = last.origins[0]
    print(f"origin {origin} (UTC), last observed {last.y0[0]:.2f} C, ridge alpha {model.alpha}")
    for h, v in enumerate(pred, start=1):
        print(f"  {origin + pd.Timedelta(hours=h)}  {v:6.2f} C")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skyfusion", description="Station + NASA POWER fusion for temperature forecasts")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("synth", help="write synthetic station and POWER files in their raw formats")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--start", default="2018-01-01")
    sp.add_argument("--end", default="2022-12-31 23:00")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("fetch-power", help="download POWER hourly data in UTC to the cache (network)")
    sp.add_argument("--start", required=True, help="YYYYMMDD")
    sp.add_argument("--end", required=True, help="YYYYMMDD")
    sp.set_defaults(func=cmd_fetch_power)

    for name, func, helptext in (("build-dataset", cmd_build, "align, check and fuse the sources into one table"),
                                 ("qc", cmd_qc, "print the quality-control report")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--station", required=True, help="OpenWeather History Bulk CSV")
        sp.add_argument("--power", nargs="*", default=[], help="POWER hourly CSV files (LST or UTC)")
        if name == "build-dataset":
            sp.add_argument("--out", default="data/fused.csv")
        sp.set_defaults(func=func)

    sp = sub.add_parser("evaluate", help="train and score all models on a time-based split, with the ablation")
    sp.add_argument("--dataset", default="data/fused.csv")
    sp.add_argument("--window", type=int)
    sp.add_argument("--horizon", type=int)
    sp.add_argument("--train-end", type=int)
    sp.add_argument("--val-end", type=int)
    sp.add_argument("--lstm", action="store_true", help="also train the LSTM (extra lstm)")
    sp.add_argument("--fast", action="store_true", help="fewer boosting iterations and LSTM epochs")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("forecast", help="forecast the next hours from the last window of a table")
    sp.add_argument("--dataset", default="data/fused.csv")
    sp.add_argument("--fused", action="store_true", help="use the fused features (default: station only)")
    sp.set_defaults(func=cmd_forecast)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
