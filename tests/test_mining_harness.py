"""Pins the mining harness (research/mining/harness.py): the agent view never
carries a hidden tier, the screen recognises a planted signal only in the
declared direction, records land in the mined ledger with the registry
schema, and full-stage adjudication applies the track-B gate."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import dsl, harness  # noqa: E402
from mining.proposal import Proposal, load_proposals  # noqa: E402
from evaluation import rulebook as rb  # noqa: E402

HIDDEN = ("virgin", "holdout", "fresh", "verdict", "family_n", "bh_threshold", "reasons")


def _proposal(cid="cand_a", expr="rank(ts_mean(returns, 5))", direction="positive"):
    return Proposal(candidate_id=cid, expression=expr, expected_direction=direction,
                    hypothesis="Short-term return persistence predicts the next month.",
                    mechanism="Under-reaction to recent news keeps prices drifting.",
                    refutation_conditions=["sign flips in holdout", "dev IC below 0"],
                    source="test")


def _screen_panel(n_syms=60, seed=0):
    """OHLCV panel spanning every evidence tier with a planted signal:
    label = 0.3 * z(rank(ts_mean(returns, 5))) + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", "2026-08-01", tz="America/New_York")
    n = len(dates)
    frames = []
    for i in range(n_syms):
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i:03d}", "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": np.exp(rng.normal(13, 0.2, n))}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    sig = dsl.compile_expression("rank(ts_mean(returns, 5))", df)
    df[harness.LABEL] = 0.3 * (sig - 0.5) + rng.normal(0, 0.05, len(df))
    df["pit"] = True
    return df


def test_agent_view_never_carries_a_hidden_tier():
    p = _proposal()
    row = harness.base_row(p, "full")
    row.update({"dev_ic": 0.02, "dev_t": 2.5, "holdout_ic": 0.03, "holdout_t": 3.1,
                "holdout_p_onesided": 0.001, "virgin_ic": 0.0, "fresh_ic": 0.0,
                "verdict": "PASS", "family_n": 3, "bh_threshold": 0.01, "reasons": ""})
    view = harness.agent_view(row)
    assert "dev_ic" in view and "dev_t" in view
    for key in view:
        assert not any(h in key for h in HIDDEN), key
    assert set(view) <= set(harness.AGENT_VISIBLE)


def test_screen_finds_planted_signal_only_in_declared_direction(tmp_path):
    panel = _screen_panel()
    ledger = tmp_path / "mined.csv"
    pos = harness.screen_one(_proposal(direction="positive"), panel, ledger)
    assert pos["screen_pass"] is True and pos["dev_t"] > harness.SCREEN_T
    assert pos["coverage"] > 0.95
    neg = harness.screen_one(_proposal(cid="cand_b", direction="negative"), panel, ledger)
    assert neg["screen_pass"] is False
    assert neg["duplicate_of"] == "cand_a", "same expression -> memory returns the old record"
    noise = harness.screen_one(_proposal(cid="cand_c", expr="rank(ts_corr(high, volume, 7))"),
                               panel, ledger)
    assert noise["screen_pass"] is False and abs(noise["dev_t"]) < 3
    bad = harness.screen_one(_proposal(cid="cand_d", expr="close.shift(-1)"), panel, ledger)
    assert bad["screen_pass"] is False and "expression:" in bad["error"]

    led = pd.read_csv(ledger)
    assert list(led.columns) == rb.MINED_COLUMNS
    assert led["stage"].tolist() == ["screen"] * 3, "the duplicate must not add a row"
    assert led["holdout_ic"].isna().all(), "screen stage must not score holdout"
    assert led.loc[0, "verdict"] == "screen_pass" and led.loc[2, "verdict"] == "error"


def test_full_adjudication_applies_track_b_gate(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2016-01-04", "2026-08-01", tz="America/New_York")
    n_names = 45
    base_rows, var_rows = [], []
    for d in dates:
        sig = rng.normal(size=n_names)
        y = 0.05 * sig + rng.normal(0, 0.05, n_names)
        pred_base = rng.normal(size=n_names)
        base_rows.append(pd.DataFrame({"date": d, "pred": pred_base, harness.LABEL: y}))
        var_rows.append(pd.DataFrame({"date": d, "pred": pred_base + 0.5 * sig,
                                      "factor_cand_a": sig, harness.LABEL: y}))
    results = tmp_path
    pd.concat(base_rows).to_parquet(results / "oos_predictions_h20_surv.parquet", index=False)
    pd.concat(var_rows).to_parquet(results / "oos_predictions_h20_mined_cand_a.parquet", index=False)

    ledger = tmp_path / "mined.csv"
    raw = pd.concat(var_rows)["factor_cand_a"].to_numpy()      # the raw expression values
    row = harness.adjudicate_full(_proposal(), "mined_cand_a", results_dir=results,
                                  family=[], ledger_path=ledger, raw_feature=raw)
    assert row["verdict"] == "PASS", row["reasons"]
    assert row["blend_dev_gain"] > 0 and row["holdout_p_onesided"] < 0.01
    assert row["holdout_n"] >= rb.HOLDOUT_MIN_SESSIONS and row["fresh_n"] > 0
    assert row["model_dev_t"] > 2 and row["model_holdout_t"] > 2
    # the same factor declared with the wrong sign fails on direction
    wrong = harness.adjudicate_full(_proposal(direction="negative"), "mined_cand_a",
                                    results_dir=results, family=[], ledger_path=ledger, raw_feature=raw)
    assert wrong["verdict"] == "FAIL" and "direction" in wrong["reasons"]
    # a NEGATIVE-direction expression: raw values anti-correlated with the label, while the
    # model score (a return prediction) is positively signed -- the raw tiers must carry the test
    neg = harness.adjudicate_full(_proposal(direction="negative"), "mined_cand_a",
                                  results_dir=results, family=[], ledger_path=ledger, raw_feature=-raw)
    assert neg["verdict"] == "PASS", neg["reasons"]
    assert neg["holdout_t"] < -2 and neg["model_holdout_t"] > 2
    # the agent view of a full record is dev-only
    view = harness.agent_view(row)
    assert "blend_dev_gain" in view and "verdict" not in view and "holdout_t" not in view


def test_prior_family_excludes_own_earlier_runs(tmp_path):
    ledger = tmp_path / "mined.csv"
    for cid, p in (("a", 0.2), ("b", 0.4), ("a", 0.1)):
        row = {c: np.nan for c in rb.MINED_COLUMNS}
        row.update({"candidate_id": cid, "stage": "full", "holdout_p_onesided": p})
        rb.append_mined(row, ledger)
    assert harness.prior_family("a", ledger) == [0.4]
    assert sorted(harness.prior_family("zzz", ledger)) == [0.1, 0.2, 0.4]


def test_load_proposals_accepts_list_and_rejects_duplicates(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('[{"candidate_id": "x_one", "expression": "rank(close)", "expected_direction": '
                 '"positive", "hypothesis": "h" , "mechanism": "m"}]', encoding="utf-8")
    ps = load_proposals(f)
    assert ps[0].candidate_id == "x_one"
    try:
        ps[0].validate()
    except ValueError as e:
        assert "hypothesis" in str(e) and "refutation" in str(e)
    else:
        raise AssertionError("thin proposal must not validate")
    f.write_text('{"proposals": [{"candidate_id": "x_one", "expression": "rank(close)", '
                 '"expected_direction": "positive", "hypothesis": "h", "mechanism": "m"}, '
                 '{"candidate_id": "x_one", "expression": "rank(open)", "expected_direction": '
                 '"positive", "hypothesis": "h", "mechanism": "m"}]}', encoding="utf-8")
    try:
        load_proposals(f)
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicates must be rejected")
