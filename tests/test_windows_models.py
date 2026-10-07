import json

import numpy as np
import pandas as pd
import pytest

from skyfusion.cli import main
from skyfusion.evaluate import add_skill, block_bootstrap_diff, run_experiment, score
from skyfusion.fuse import feature_sets
from skyfusion.models import Climatology, GBMDirect, Persistence, RidgeDirect, SeasonalNaive
from skyfusion.windows import make_samples, split_by_year

W, H = 24, 6


@pytest.fixture(scope="module")
def samples(fused):
    df, _ = fused
    return make_samples(df, feature_sets(df)["station"], W, H)


@pytest.fixture(scope="module")
def split(samples):
    return split_by_year(samples, 2018, 2019, H)


def test_targets_are_the_future_of_each_origin(fused, samples):
    # Problems 5 and 6: target h of an origin is the temperature h hours LATER.
    df, _ = fused
    for k in (0, 500, len(samples) - 1):
        t = samples.origins[k]
        for h in (1, H):
            assert samples.Y[k, h - 1] == df.target_temp_c.loc[t + pd.Timedelta(hours=h)]
        assert samples.y0[k] == df.target_temp_c.loc[t]
        col = samples.features.index("st_temp_c")
        assert samples.X_seq[k, -1, col] == df.st_temp_c.loc[t] or np.isnan(df.st_temp_c.loc[t])


def test_samples_skip_missing_targets(fused, samples):
    df, _ = fused
    assert not np.isnan(samples.Y).any() and not np.isnan(samples.y0).any()
    assert len(samples) < len(df) - W - H


def test_irregular_grid_rejected(fused):
    df, _ = fused
    with pytest.raises(ValueError, match="hourly grid"):
        make_samples(df.drop(df.index[5]), ["st_temp_c"], W, H)


def test_split_by_year_with_embargo(split):
    # Problem 2: the validation set is not empty, and no target hour crosses a boundary.
    assert len(split.train) and len(split.val) and len(split.test)
    assert (split.train.origins + pd.Timedelta(hours=H)).year.max() == 2018
    assert split.val.origins.year.min() == 2019 and (split.val.origins + pd.Timedelta(hours=H)).year.max() == 2019
    assert split.test.origins.year.min() == 2020


def test_empty_split_raises(samples):
    with pytest.raises(ValueError, match="empty split"):
        split_by_year(samples, 2020, 2021, H)


def test_baselines(split):
    t = split.test
    assert np.array_equal(Persistence().predict(t)[:, 3], t.y0)
    sn = SeasonalNaive().predict(t)
    ok = ~np.isnan(t.y_hist[:, 0])
    assert np.allclose(sn[ok, 0], t.y_hist[ok, 0])  # y(t+1-24) is the first hour of a 24 h window
    clim = Climatology().fit(split.train).predict(t)
    assert clim.shape == t.Y.shape and np.isfinite(clim).all()


def test_ridge_scaling_fit_on_train_only_and_alpha_on_val(split):
    # Problem 9: scaling inside the pipeline, fit on train samples only.
    m = RidgeDirect().fit(split.train, split.val)
    imputed = m.pipe.named_steps["impute"].transform(split.train.flat())
    assert np.allclose(m.pipe.named_steps["scale"].mean_, imputed.mean(axis=0))
    assert m.alpha in (0.1, 1.0, 10.0, 100.0)


def test_learned_models_beat_persistence(split):
    pers = score(Persistence().predict(split.test), split.test, "persistence", "station")
    ridge = score(RidgeDirect().fit(split.train, split.val).predict(split.test), split.test, "ridge", "station")
    gbm_model = GBMDirect(max_iter=30).fit(split.train, split.val)
    gbm = score(gbm_model.predict(split.test), split.test, "gbm", "station")
    assert len(gbm_model.best_iter) == H
    rows = add_skill([pers, ridge, gbm])
    assert rows[0].skill.tolist() == [0.0] * H
    assert ridge.rmse[0] < pers.rmse[0] and (ridge.skill > 0).all() and (gbm.skill > 0).all()


def test_each_model_is_scored_on_its_own_predictions(split):
    # Problem 4: the metric comes from the model's own output.
    t = split.test
    pred = Persistence().predict(t)
    s = score(pred, t, "persistence", "station")
    assert s.mae[0] == pytest.approx(np.abs(t.y0 - t.Y[:, 0]).mean())
    with pytest.raises(ValueError):
        score(pred[:, :2], t, "x", "station")


def test_block_bootstrap(split):
    t = split.test
    p = Persistence().predict(t)
    d, lo, hi = block_bootstrap_diff(p, p + 0.0, t, 1, n_boot=50)
    assert d == 0 and lo == 0 and hi == 0
    d2, lo2, hi2 = block_bootstrap_diff(p + 1.0, p, t, 1, n_boot=50)
    assert d2 > 0 and lo2 <= d2 <= hi2


def test_experiment_ablation_uses_identical_samples(fused):
    # Problem 3: the single-source and fused runs predict the same target on the same hours.
    df, _ = fused
    exp = run_experiment(df, W, H, 2018, 2019, fast=True)
    assert exp.splits["station"].test.origins.equals(exp.splits["fused"].test.origins)
    assert np.array_equal(exp.splits["station"].test.Y, exp.splits["fused"].test.Y)
    models = {(r.model, r.features) for r in exp.rows}
    assert ("persistence", "station") in models and ("ridge", "fused") in models and ("persistence", "fused") not in models
    assert {a["model"] for a in exp.ablation} == {"ridge", "gbm"}


def test_lstm_direct(split):
    pytest.importorskip("torch")
    from skyfusion.lstm import LSTMDirect

    small_train = split.train.subset(np.arange(len(split.train)) < 2000)
    small_val = split.val.subset(np.arange(len(split.val)) < 500)
    m = LSTMDirect(hidden=8, epochs=2, seed=0).fit(small_train, small_val)
    assert len(m.history) >= 1
    assert m.predict(split.test).shape == split.test.Y.shape


def test_cli_end_to_end(tmp_path, capsys):
    d = tmp_path / "syn"
    assert main(["synth", "--out", str(d), "--start", "2018-01-01", "--end", "2020-03-31 23:00", "--seed", "5"]) == 0
    st, pw = d / "station_openweather.csv", d / "power_lst.csv"
    assert main(["qc", "--station", str(st), "--power", str(pw)]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["target_missing"] > 0
    out = tmp_path / "fused.csv"
    assert main(["build-dataset", "--station", str(st), "--power", str(pw), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--dataset", str(out), "--horizon", "6", "--train-end", "2018", "--val-end", "2019",
                 "--fast", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["rows"][0]["mae"]) == 6
    assert main(["evaluate", "--dataset", str(out), "--horizon", "2", "--train-end", "2018", "--val-end", "2019", "--fast"]) == 0
    assert "ablation" in capsys.readouterr().out


def test_cli_forecast(tmp_path, capsys, monkeypatch):
    d = tmp_path / "syn"
    main(["synth", "--out", str(d), "--start", "2018-01-01", "--end", "2020-03-31 23:00", "--seed", "6"])
    out = tmp_path / "fused.csv"
    main(["build-dataset", "--station", str(d / "station_openweather.csv"), "--power", str(d / "power_lst.csv"),
          "--out", str(out)])
    monkeypatch.setenv("SKYFUSION_TRAIN_END", "2018")
    monkeypatch.setenv("SKYFUSION_VAL_END", "2019")
    monkeypatch.setenv("SKYFUSION_HORIZON", "4")
    capsys.readouterr()
    assert main(["forecast", "--dataset", str(out), "--fused"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("origin") and len(lines) == 5
