"""Pins Track P (research/mining/pool.py): signed-rank composite, the
minimum-fraction rule, admission (screen pass, no oracle flag, not
incumbent-redundant, correlation with members below 0.7, residual t against
the composite at least 1.5), idempotent admission, and status/checkpoints."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import pool as pl  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_pool_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "POOL_DIR", tmp_path)


def _panel(n_dates=60, n_syms=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates, tz="America/New_York")
    rows = []
    for d in dates:
        for i in range(n_syms):
            rows.append({"date": d, "symbol": f"S{i:02d}", "close": 10 + rng.normal(), "volume": 1000 + rng.integers(0, 100),
                         "open": 10.0, "high": 11.0, "low": 9.0})
    df = pd.DataFrame(rows)
    df["future_return_20d"] = rng.normal(size=len(df))
    df["pit"] = True
    return df


def test_signed_rank_and_composite_rules():
    dates = np.array(["a", "a", "a", "b", "b", "b"])
    x = np.array([3.0, 1.0, 2.0, 1.0, np.nan, 3.0])
    r = pl.signed_rank(dates, x, -1.0)
    assert r[0] < r[2] < r[1]                                    # sign flipped: largest x -> most negative
    assert np.isnan(r[4])
    f = pd.DataFrame({"m1": [0.2, np.nan, np.nan], "m2": [0.4, 0.1, np.nan], "m3": [np.nan, 0.3, np.nan]})
    c = pl.composite(f)
    assert abs(c[0] - 0.3) < 1e-12 and abs(c[1] - 0.2) < 1e-12 and np.isnan(c[2])
    assert np.isnan(pl.composite(pd.DataFrame(index=[0, 1]))).all()


def test_member_features_cache_and_pool_signal():
    df = _panel()
    members = [{"hash": "h1", "expression": "rank(close)", "expected_direction": "positive"},
               {"hash": "h2", "expression": "rank(volume)", "expected_direction": "negative"}]
    f1 = pl.member_features(df, members, 20)
    assert list(f1.columns) == ["h1", "h2"] and pl.features_path(20).exists()
    f2 = pl.member_features(df, members, 20)                    # from cache
    np.testing.assert_allclose(f1.to_numpy(), f2.to_numpy(), equal_nan=True)
    pool = {"horizon": 20, "members": members, "releases": []}
    sig = pl.pool_signal(df, 20, pool)
    assert np.isfinite(sig).mean() > 0.9
    assert np.isnan(pl.pool_signal(df, 20, {"horizon": 20, "members": [], "releases": []})).all()


def _row(cid, **kw):
    base = {"candidate_id": cid, "expression": "rank(close)", "canonical": "rank(close)", "expected_direction": "positive",
            "screen_pass": True, "quarantined": False, "duplicate_of": "", "redundant_with": "",
            "pool_corr_max": 0.1, "pool_corr_with": "", "residual_vs_pool_t": 2.0, "pool_size": 3,
            "dev_ic": 0.01, "dev_t": 2.5}
    base.update(kw)
    return base


def test_admission_rules():
    assert pl.admissible(_row("a"))[0]
    assert not pl.admissible(_row("b", screen_pass=False))[0]
    assert not pl.admissible(_row("c", quarantined=True))[0]
    assert not pl.admissible(_row("d", redundant_with="incumbent:returns_20d"))[0]
    assert not pl.admissible(_row("e", pool_corr_max=0.75))[0]
    assert not pl.admissible(_row("f", residual_vs_pool_t=1.2))[0]
    assert pl.admissible(_row("g", pool_size=0, residual_vs_pool_t=float("nan")))[0]   # first member: no composite yet
    assert pl.admissible(_row("h", redundant_with="some_prior"))[0]                    # prior-redundant is fine for the pool


def test_admit_is_idempotent_and_status_reports_checkpoints():
    rows = [_row("a"), _row("b", expression="rank(volume)", canonical="rank(volume)", expected_direction="negative"),
            _row("c", expression="rank(close)", canonical="rank(close)")]                # same expression as a
    new = pl.admit(20, rows, source="run1")
    assert [m["candidate_id"] for m in new] == ["a", "b"]
    assert pl.admit(20, rows, source="run1") == []
    st = pl.status(20)
    assert st["n_members"] == 2 and st["next_checkpoint"] == 25 and not st["due"]
    assert st["by_source"] == {"run1": 2}


def test_assess_against_pool_signed_residual():
    df = _panel(n_dates=80)
    dates = pd.to_datetime(df["date"]).to_numpy()
    label = df["future_return_20d"].to_numpy()
    members = [{"hash": "h1", "expression": "rank(close)", "expected_direction": "positive"}]
    feats = pl.member_features(df, members, 20, use_cache=False)
    comp = pl.composite(feats)
    cand = df["close"].to_numpy(dtype=float)                       # identical to the member -> corr ~ 1
    out = pl.assess_against_pool(cand, 1.0, dates, label, feats, comp, nw_lags=19)
    assert out["pool_size"] == 1 and out["pool_corr_max"] > 0.99 and out["pool_corr_with"] == "h1"
    assert abs(out["residual_vs_pool_t"]) < 1e-9                    # rank-identical: nothing left
    empty = pl.assess_against_pool(cand, 1.0, dates, label, pd.DataFrame(index=df.index), comp, nw_lags=19)
    assert empty["pool_size"] == 0 and np.isnan(empty["residual_vs_pool_t"])
