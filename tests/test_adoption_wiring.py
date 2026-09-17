"""Adoption wiring (2026-09-17): the registry's probation weight cap and the
IC-monitor removal trigger are consumed by production, and the pool
feature cache is keyed by a panel fingerprint."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "research"))

from factors import factor_weighting as fw  # noqa: E402
from factors import mined_factors as mf  # noqa: E402


def test_caps_and_active_set_renormalise():
    w = {"a": 0.5, "mined_x": 0.3, "b": 0.2}
    out = fw.apply_caps_and_active_set(w, active={"a", "mined_x", "b"}, caps={"mined_x": 0.025})
    assert out["mined_x"] == pytest.approx(0.025)
    assert out["a"] == pytest.approx(0.5 / 0.7 * 0.975) and out["b"] == pytest.approx(0.2 / 0.7 * 0.975)
    assert sum(out.values()) == pytest.approx(1.0)
    # a factor no longer in the active set (removed from the registry) drops out, the rest renormalise
    out2 = fw.apply_caps_and_active_set(w, active={"a", "b"}, caps={})
    assert set(out2) == {"a", "b"} and out2["a"] == pytest.approx(0.5 / 0.7)
    # a cap that is not binding changes nothing; two caps resolve together
    assert fw.apply_caps_and_active_set(w, caps={"mined_x": 0.5}) == pytest.approx(w)
    out3 = fw.apply_caps_and_active_set({"a": 0.4, "p": 0.3, "q": 0.3}, caps={"p": 0.05, "q": 0.1})
    assert out3 == pytest.approx({"a": 0.85, "p": 0.05, "q": 0.1})


def test_finalize_weights_reads_registry_caps_and_active_groups(monkeypatch):
    import factors.factor_definitions as fd
    monkeypatch.setattr(fd, "FACTOR_GROUPS", {"a": [], "b": [], "mined_x": []})
    monkeypatch.setattr(mf, "probation_caps", lambda path=None: {"mined_x": 0.025})
    out = fw.finalize_weights({"a": 0.5, "mined_x": 0.3, "b": 0.2, "gone": 0.1}, log=False)
    assert "gone" not in out and out["mined_x"] == pytest.approx(0.025) and sum(out.values()) == pytest.approx(1.0)


def test_ic_monitor_triggers_remove_probation_and_flag_structural(tmp_path):
    reg = tmp_path / "mined_factors.json"
    reg.write_text(json.dumps({"adopted": [
        {"id": "px", "expression": "rank(close)", "expected_direction": "positive", "horizon": 20,
         "adoption_tier": "probation", "weight_cap": 0.025},
        {"id": "sx", "expression": "rank(volume)", "expected_direction": "positive", "horizon": 20,
         "adoption_tier": "structural"},
        {"id": "ok", "expression": "rank(open)", "expected_direction": "positive", "horizon": 20,
         "adoption_tier": "probation", "weight_cap": 0.025},
    ]}), encoding="utf-8")
    mon = tmp_path / "ic_monitor_latest.json"
    mon.write_text(json.dumps({"factors": {"factor_px": {"status": "WARN", "as_of": "2026-09-13"},
                                           "factor_sx": {"status": "ALERT", "as_of": "2026-09-13"},
                                           "factor_ok": {"status": "OK", "as_of": "2026-09-13"}}}), encoding="utf-8")
    assert mf.probation_caps(reg) == {"px": 0.025, "ok": 0.025}
    changed = mf.apply_ic_monitor_triggers(reg, mon, today="2026-09-14")
    assert [e["id"] for e in changed] == ["px", "sx"]
    data = json.loads(reg.read_text(encoding="utf-8"))["adopted"]
    px, sx, ok = data
    assert px["status"] == "removed" and px["removed_on"] == "2026-09-14" and "WARN" in px["removal_reason"]
    assert sx.get("status", "production") == "production" and "ALERT" in sx["review_flag"]
    assert "status" not in ok
    # consumers: the removed factor is gone, its cap with it; idempotent
    assert [e["id"] for e in mf.load_adopted(reg)] == ["sx", "ok"] and mf.probation_caps(reg) == {"ok": 0.025}
    assert mf.apply_ic_monitor_triggers(reg, mon, today="2026-09-15") == []
    assert mf.apply_ic_monitor_triggers(tmp_path / "missing.json", mon) == []


def _panel(n_dates=30, n_syms=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"date": d, "symbol": f"S{j:02d}", "close": 10 + rng.normal(), "volume": 1000 + rng.integers(0, 100)}
            for d in pd.bdate_range("2026-01-05", periods=n_dates) for j in range(n_syms)]
    return pd.DataFrame(rows)


def test_pool_feature_cache_is_keyed_by_panel_fingerprint(tmp_path, monkeypatch):
    from mining import pool as pl
    monkeypatch.setattr(pl, "POOL_DIR", tmp_path)
    df = _panel()
    members = [{"hash": "h1", "expression": "rank(close)", "expected_direction": "positive"}]
    f1 = pl.member_features(df, members, 20)
    cache = pl.features_path(20)
    assert cache.exists() and cache.with_suffix(".meta.json").exists()
    fp = json.loads(cache.with_suffix(".meta.json").read_text())["fingerprint"]
    assert fp == pl.panel_fingerprint(df)
    # tamper with the cached values: the SAME panel reads them back (reuse) ...
    tampered = pd.read_parquet(cache)
    tampered["h1"] = 99.0
    tampered.to_parquet(cache, index=False)
    f2 = pl.member_features(df, members, 20)
    assert (f2["h1"] == 99.0).all()
    # ... a re-sorted panel of the same length does NOT (different fingerprint -> recomputed, cache untouched)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert pl.panel_fingerprint(shuffled) != fp and len(shuffled) == len(df)
    f3 = pl.member_features(shuffled, members, 20)
    assert not (f3["h1"] == 99.0).any()
    assert (pd.read_parquet(cache)["h1"] == 99.0).all()          # a foreign panel never overwrites the cache
    np.testing.assert_allclose(np.sort(f3["h1"].dropna().to_numpy()), np.sort(f1["h1"].dropna().to_numpy()))
