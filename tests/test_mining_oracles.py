"""Pins the process-level oracles (research/mining/oracles.py): each fires on
its injected defect and stays silent on a clean candidate; the harness
quarantines a flagged candidate, refuses it at the full stage, and refuses
to emit a view that carries hidden information."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import dsl, harness, oracles  # noqa: E402
from mining.proposal import Proposal  # noqa: E402
from evaluation.metrics import daily_rank_ic  # noqa: E402

RNG = np.random.default_rng(5)


def _panel(n_syms=50):
    dates = pd.bdate_range("2021-01-04", "2026-08-01", tz="America/New_York")
    n = len(dates)
    frames = []
    for i in range(n_syms):
        close = 50 * np.exp(np.cumsum(RNG.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"date": dates, "symbol": f"S{i:03d}", "open": close, "high": close * 1.01,
                                    "low": close * 0.99, "close": close, "volume": np.exp(RNG.normal(13, 0.3, n))}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    df = harness.add_label(df)
    df["pit"] = True
    return df


def test_future_perturbation_is_silent_on_causal_and_fires_on_leaky_producers():
    df = _panel(20)
    assert oracles.future_perturbation("rank(ts_mean(returns, 5))", df, n_syms=10)["fired"] is False
    leaky = lambda e, d: dsl.compile_expression(e, d).groupby(d["symbol"]).shift(-3)      # noqa: E731
    assert oracles.future_perturbation("rank(ts_mean(returns, 5))", df, compile_fn=leaky, n_syms=10)["fired"] is True
    centred = lambda e, d: dsl.compile_expression(e, d).groupby(d["symbol"]).transform(  # noqa: E731
        lambda g: g.rolling(5, center=True, min_periods=1).mean())
    assert oracles.future_perturbation("close", df, compile_fn=centred, n_syms=10)["fired"] is True


def test_strength_and_visibility_oracles():
    assert oracles.strength(0.02, 2.5)["fired"] is False
    assert oracles.strength(0.12, 3.0)["fired"] is True
    assert oracles.strength(0.03, 7.0)["fired"] is True
    row = {"candidate_id": "x", "dev_t": 2.1, "holdout_t": 3.3333, "verdict": "PASS", "reasons": ""}
    ok = {"candidate_id": "x", "dev_t": 2.1}
    assert oracles.visibility(ok, row, ("candidate_id", "dev_t"))["fired"] is False
    bad_key = {"candidate_id": "x", "dev_t": 2.1, "holdout_t": 3.3333}
    assert oracles.visibility(bad_key, row, ("candidate_id", "dev_t"))["bad_keys"] == ["holdout_t"]
    smuggled = {"candidate_id": "x", "note": "fyi 3.3333"}
    assert oracles.visibility(smuggled, row, ("candidate_id", "note"))["leaked_values"] == ["holdout_t"]


def test_membership_oracle_fires_when_the_official_mask_is_wrong(tmp_path, monkeypatch):
    df = _panel(30)
    monkeypatch.setattr(oracles, "PIT_INDEPENDENT", tmp_path / "pit.parquet")
    truth = df["symbol"] != "S000"          # S000 is never a member
    monkeypatch.setattr(oracles, "independent_pit_mask", lambda panel, cache=None, **kw: truth.to_numpy())
    feat = dsl.compile_expression("rank(ts_mean(returns, 5))", df).to_numpy()
    dev = lambda d: harness.segment(d, "seen_dev")                                        # noqa: E731
    sub = pd.DataFrame({"date": df["date"], harness.LABEL: df[harness.LABEL], "_f": feat})[truth]
    official = float(daily_rank_ic(dev(sub), "_f", target_col=harness.LABEL, min_names_per_date=20).mean())
    ok = oracles.membership(df, feat, harness.LABEL, dev, official, daily_rank_ic, 20)
    assert ok["fired"] is False
    wrong = float(daily_rank_ic(dev(pd.DataFrame({"date": df["date"], harness.LABEL: df[harness.LABEL], "_f": feat})),
                                "_f", target_col=harness.LABEL, min_names_per_date=20).mean())   # all rows
    assert oracles.membership(df, feat, harness.LABEL, dev, wrong, daily_rank_ic, 20)["fired"] is True


def test_controls_fire_on_a_misaligned_label():
    df = _panel(40)
    dev = lambda d: harness.segment(d, "seen_dev")                                        # noqa: E731
    clean = oracles.controls(df, harness.LABEL, dev, daily_rank_ic, 20)
    assert clean["fired"] is False, clean
    bad = df.copy()
    prev = bad.groupby("symbol")["close"].shift(1)
    fwd = bad.groupby("symbol")["close"].shift(-harness.HORIZON)
    bad[harness.LABEL] = np.log(fwd / prev)                      # today's return leaks into the label
    assert oracles.controls(bad, harness.LABEL, dev, daily_rank_ic, 20)["controls"]["returns"]["fired"] is True


def test_harness_quarantines_and_refuses_a_leaky_candidate(tmp_path, monkeypatch):
    df = _panel(40)
    ledger = tmp_path / "mined.csv"
    monkeypatch.setattr(oracles, "independent_pit_mask", lambda panel, cache=None, **kw: panel["pit"].to_numpy(dtype=bool))
    monkeypatch.setattr(harness, "_incumbent_refs_for", lambda p: {})
    orig = dsl.compile_expression
    monkeypatch.setattr(dsl, "compile_expression", lambda e, d, **k: orig(e, d, **k).groupby(d["symbol"]).shift(-5))
    p = Proposal(candidate_id="leaky", expression="rank(ts_mean(returns, 5))", expected_direction="positive",
                 hypothesis="A probe that a leaky producer will inflate.", mechanism="None; harness self-test.",
                 refutation_conditions=["x"], mechanism_tag="probe", source="test")
    views = harness.screen([p], df, ledger, record=True, run_controls=False)
    v = views[0]
    assert v["quarantined"] is True and "future_leak" in json.loads(v["oracle_flags"])
    for k in v:
        assert k in harness.AGENT_VISIBLE
    with pytest.raises(SystemExit, match="quarantined"):
        harness.assert_representative("leaky", ledger)
    harness.assert_representative("leaky", ledger, force=True)


def test_harness_refuses_to_emit_hidden_information(tmp_path, monkeypatch):
    df = _panel(30)
    ledger = tmp_path / "mined.csv"
    monkeypatch.setattr(oracles, "independent_pit_mask", lambda panel, cache=None, **kw: panel["pit"].to_numpy(dtype=bool))
    monkeypatch.setattr(harness, "_incumbent_refs_for", lambda p: {})
    monkeypatch.setattr(harness, "AGENT_VISIBLE", harness.AGENT_VISIBLE + ("holdout_t",))
    orig = harness.screen_one

    def leaky(p, panel, ledger_path=None, record=True):
        row = orig(p, panel, ledger_path, record)
        row["holdout_t"] = 3.21
        return row
    monkeypatch.setattr(harness, "screen_one", leaky)
    p = Proposal(candidate_id="exposed", expression="rank(close)", expected_direction="positive",
                 hypothesis="A probe under an exposure mutation.", mechanism="None; harness self-test.",
                 refutation_conditions=["x"], mechanism_tag="probe", source="test")
    with pytest.raises(RuntimeError, match="visibility oracle"):
        harness.screen([p], df, ledger, record=True, run_controls=False)
