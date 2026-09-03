"""Pins screen-stage redundancy clustering (research/mining/redundancy.py):
look-alike passes form one cluster with one representative, a pass that
restates an incumbent feature is never a representative, residual t drops
for a look-alike and survives for independent information, and the harness
refuses a non-representative at the full stage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import redundancy as rd, harness, dsl  # noqa: E402
from mining.proposal import Proposal  # noqa: E402
from evaluation import rulebook as rb  # noqa: E402

RNG = np.random.default_rng(11)


def _panel(n_syms=60):
    dates = pd.bdate_range("2021-01-04", "2026-08-01", tz="America/New_York")
    n = len(dates)
    frames = []
    for i in range(n_syms):
        close = 50 * np.exp(np.cumsum(RNG.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"date": dates, "symbol": f"S{i:03d}", "open": close, "high": close * 1.01,
                                    "low": close * 0.99, "close": close, "volume": np.exp(RNG.normal(13, 0.4, n))}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    sig = dsl.compile_expression("rank(ts_max(volume, 20) / ts_mean(volume, 20))", df)
    df[harness.LABEL] = -0.3 * (sig - 0.5) + RNG.normal(0, 0.05, len(df))
    df["pit"] = True
    return df


def test_mean_rank_corr_and_residual_behave():
    dates = np.repeat(np.arange(200), 60)
    a = RNG.normal(size=len(dates))
    b = a + 0.1 * RNG.normal(size=len(dates))
    c = RNG.normal(size=len(dates))
    assert rd.mean_rank_corr(dates, a, b) > 0.95
    assert abs(rd.mean_rank_corr(dates, a, c)) < 0.1
    y = a + RNG.normal(size=len(dates))
    ic_raw = rd.residual_rank_ic(dates, a, np.zeros(len(dates)), y).mean()   # constant ref: raw IC
    ic_self = rd.residual_rank_ic(dates, a, b, y).mean()         # a explained by its look-alike b
    ic_indep = rd.residual_rank_ic(dates, a, c, y).mean()        # a orthogonalised against noise
    assert ic_raw > 0.5 and abs(ic_self) < 0.5 * ic_raw and ic_indep > 0.9 * ic_raw


def test_assess_clusters_lookalikes_and_flags_incumbent_restatements():
    dates = np.repeat(np.arange(300), 60)
    n = len(dates)
    base = RNG.normal(size=n)
    y = -base + RNG.normal(size=n)
    passes = [
        {"candidate_id": "spike_a", "dev_t": -2.5, "x": base + 0.05 * RNG.normal(size=n)},
        {"candidate_id": "spike_b", "dev_t": -3.1, "x": base * 2 + 0.05 * RNG.normal(size=n)},
        {"candidate_id": "indep", "dev_t": -2.1, "x": RNG.normal(size=n)},
        {"candidate_id": "vol_copy", "dev_t": -2.2, "x": None},
    ]
    incumbent = RNG.normal(size=n)
    passes[3]["x"] = incumbent + 0.05 * RNG.normal(size=n)
    out = rd.assess(passes, dates, y, refs={"volume_std_20d": incumbent}, priors={})
    assert out["spike_a"]["cluster_id"] == "spike_b" and out["spike_b"]["cluster_rep"] is True
    assert out["spike_a"]["cluster_rep"] is False and out["spike_a"]["redundant_with"] == "spike_b"
    assert out["spike_a"]["cluster_size"] == 2
    assert out["indep"]["cluster_rep"] is True and out["indep"]["redundant_with"] == ""
    assert out["vol_copy"]["cluster_rep"] is False
    assert out["vol_copy"]["redundant_with"] == "incumbent:volume_std_20d"
    from evaluation.metrics import summarize_ic
    raw_t = summarize_ic(rd.residual_rank_ic(dates, passes[0]["x"], np.zeros(n), y), nw_lags=19)["t_stat"]
    assert abs(out["spike_a"]["residual_dev_t"]) < 0.5 * abs(raw_t)         # look-alike removed most of it
    assert out["spike_a"]["residual_vs"] == "batch:spike_b"
    assert out["spike_b"]["residual_vs"] == "batch:spike_a"                   # seen from both sides
    assert out["indep"]["residual_dev_t"] == passes[2]["dev_t"]              # nothing to remove


def test_prior_representative_keeps_precedence():
    dates = np.repeat(np.arange(300), 60)
    n = len(dates)
    base = RNG.normal(size=n)
    y = RNG.normal(size=n)
    passes = [{"candidate_id": "new_one", "dev_t": -4.0, "x": base + 0.05 * RNG.normal(size=n)}]
    out = rd.assess(passes, dates, y, refs={}, priors={"old_rep": base})
    assert out["new_one"]["cluster_rep"] is False and out["new_one"]["redundant_with"] == "old_rep"


def test_harness_screen_marks_clusters_and_full_refuses_non_representative(tmp_path, monkeypatch):
    panel = _panel()
    ledger = tmp_path / "mined.csv"
    monkeypatch.setattr(rd, "REF_CACHE", tmp_path / "ref.parquet")
    monkeypatch.setattr(rd, "PASS_CACHE", tmp_path / "pass.parquet")
    monkeypatch.setattr(rd, "PASS_CACHE_META", tmp_path / "pass.json")
    monkeypatch.setattr(harness, "_incumbent_refs_for", lambda p: {})   # synthetic panel has no incumbents

    def prop(cid, expr):
        return Proposal(candidate_id=cid, expression=expr, expected_direction="negative",
                        hypothesis="Volume spikes are followed by lower returns next month.",
                        mechanism="Attention-driven buying reverses after the event.",
                        refutation_conditions=["x"], source="test")

    views = harness.screen([prop("spike20", "ts_max(volume, 20) / ts_mean(volume, 20)"),
                            prop("spike20b", "rank(ts_max(volume, 20) / ts_mean(volume, 20))"),
                            prop("noise", "rank(ts_corr(open, high, 7))")], panel, ledger)
    by = {v["candidate_id"]: v for v in views}
    assert by["spike20"]["screen_pass"] and by["spike20b"]["screen_pass"]
    reps = [c for c in ("spike20", "spike20b") if by[c]["cluster_rep"]]
    assert len(reps) == 1, "one representative per cluster"
    other = [c for c in ("spike20", "spike20b") if c not in reps][0]
    assert by[other]["redundant_with"] == reps[0] and by[other]["max_corr"] > 0.9
    # a candidate must never be its own prior (its ledger row predates clustering)
    for cid, v in by.items():
        assert not str(v.get("corr_with", "")).endswith(":" + cid), (cid, v.get("corr_with"))
    assert by[other]["corr_with"] == "batch:" + reps[0]
    # rank(f) is rank-identical to f: orthogonalising leaves exactly nothing
    assert by[other]["residual_dev_t"] == 0.0 and by[reps[0]]["residual_dev_t"] == 0.0
    assert "cluster_rep" not in by["noise"]                    # non-pass: nothing to cluster

    led = pd.read_csv(ledger)
    row = led[led["candidate_id"] == other].iloc[-1]
    assert rb.truthy(row["cluster_rep"]) is False and row["redundant_with"] == reps[0]
    with pytest.raises(SystemExit, match="not the representative"):
        harness.assert_representative(other, ledger)
    harness.assert_representative(reps[0], ledger)              # representative passes
    harness.assert_representative(other, ledger, force=True)    # human override
