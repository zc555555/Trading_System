"""strategy/selection.py: the production selection policy, pinned step by
step against the arithmetic the signal script used inline until 2026-09-14
(review item 3: evaluation must replay production)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "research"))

from strategy import selection as sel  # noqa: E402

P = sel.SelectionParams()


# 1. smoothing --------------------------------------------------------------
def test_smoothing_reproduces_the_production_day_weight_rule():
    dates = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08"])
    rows = [{"date": d, "symbol": "A", "factor_x": v} for d, v in zip(dates, [1, 2, 3, 4, 5])]
    rows += [{"date": dates[3], "symbol": "B", "factor_x": 10}, {"date": dates[4], "symbol": "B", "factor_x": 20}]
    rows += [{"date": dates[4], "symbol": "C", "factor_x": 7}]
    rows += [{"date": dates[0], "symbol": "D", "factor_x": 99}]      # older than the window's oldest? no: it IS in it
    out = sel.smooth_scores(pd.DataFrame(rows), ["factor_x"], P).set_index("symbol")["factor_x"]
    assert out["A"] == pytest.approx(0.10 * 1 + 0.15 * 2 + 0.20 * 3 + 0.25 * 4 + 0.30 * 5)   # 3.5
    assert out["B"] == pytest.approx(20 * 0.6 + 10 * 0.4)            # last two weights (.15,.10) renormalised, newest first
    assert out["C"] == pytest.approx(7.0)
    assert out["D"] == pytest.approx(99.0)                            # one session available -> full weight
    # a NaN inside the window kills the prediction (never silently skipped)
    rows2 = rows + [{"date": dates[2], "symbol": "C", "factor_x": np.nan}]
    out2 = sel.smooth_scores(pd.DataFrame(rows2), ["factor_x"], P).set_index("symbol")["factor_x"]
    assert np.isnan(out2["C"]) and out2["A"] == pytest.approx(3.5)
    # only the last smooth_days sessions count
    rows3 = rows + [{"date": pd.Timestamp("2026-08-25"), "symbol": "A", "factor_x": 1000}]
    out3 = sel.smooth_scores(pd.DataFrame(rows3), ["factor_x"], P).set_index("symbol")["factor_x"]
    assert out3["A"] == pytest.approx(3.5)
    # smoothing off = the latest score
    off = sel.smooth_scores(pd.DataFrame(rows), ["factor_x"], sel.SelectionParams(smooth_days=1)).set_index("symbol")["factor_x"]
    assert off["A"] == 5 and off["B"] == 20


def test_smooth_panel_matches_smooth_scores_session_by_session():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2026-01-05", periods=12)
    rows = []
    for i, d in enumerate(dates):
        for s in "ABC":
            if s == "B" and i in (4, 7):
                continue                                              # B misses two sessions
            rows.append({"date": d, "symbol": s, "pred": rng.normal()})
    panel = pd.DataFrame(rows)
    got = sel.smooth_panel(panel, "pred", P)
    for d in dates[2:]:
        window = panel[panel["date"] <= d]
        want = sel.smooth_scores(window, ["pred"], P).set_index("symbol")["pred"]
        here = panel["date"] == d
        for s in panel.loc[here, "symbol"]:
            g = got[here & (panel["symbol"] == s)].iloc[0]
            assert g == pytest.approx(want[s]), (d, s)


class _Model:
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def test_smoothing_and_blend_equal_the_signal_script(tmp_path, monkeypatch):
    """The refactor must not change tonight's numbers: the script's
    get_weighted_predictions_multi_factor == z-score -> smooth -> blend."""
    gds = pytest.importorskip("get_daily_signals_multi_factor")
    monkeypatch.setattr(gds, "FACTOR_WEIGHTS", {"fa": 0.6, "fb": 0.4})
    monkeypatch.setattr(gds, "Path", lambda *a, **k: tmp_path / "gds.py")
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2026-08-20", periods=7)
    rows = []
    for i, d in enumerate(dates):
        for s in ["AAA", "BBB", "CCC", "DDD"]:
            if s == "DDD" and i < 5:
                continue                                              # DDD only has the last two sessions
            rows.append({"date": d, "symbol": s, "f1": rng.normal(), "f2": rng.normal(), "close": 100.0})
    df = pd.DataFrame(rows)
    ens = {"fa": {"models": {"m": _Model()}, "weights": {"m": 1.0}, "feature_cols": ["f1", "f2"]},
           "fb": {"models": {"m": _Model()}, "weights": {"m": 1.0}, "feature_cols": ["f2", "f1"]}}
    script = gds.get_weighted_predictions_multi_factor(df, ens, n_days=5).set_index("symbol")["prediction"]
    scores = df[["date", "symbol"]].copy()
    for col, feat in (("factor_fa", "f1"), ("factor_fb", "f2")):        # the script's own per-date z-score
        scores[col] = np.nan
        for d in dates:
            idx = df.index[df["date"] == d]
            scores.loc[idx, col] = gds._per_date_zscore(df.loc[idx, feat].to_numpy())
    sm = sel.smooth_scores(scores, ["factor_fa", "factor_fb"], P)
    sm["prediction"] = sel.blend(sm, {"fa": 0.6, "fb": 0.4})
    mine = sm.set_index("symbol")["prediction"]
    for s in script.index:                                                # float32 model inputs in the script
        assert mine[s] == pytest.approx(script[s], rel=1e-5, abs=1e-7), s


# 3. trend filter ------------------------------------------------------------
def _production_trend_loop(df):
    """Verbatim port of the signal script's per-symbol loop (pre-refactor)."""
    out = []
    for symbol in df["symbol"].unique():
        symbol_data = df[df["symbol"] == symbol].sort_values("date")
        if len(symbol_data) < 30:
            continue
        recent_30 = symbol_data.tail(30)
        current_price = recent_30["close"].iloc[-1]
        price_7d_ago = recent_30["close"].iloc[-7]
        price_14d_ago = recent_30["close"].iloc[-14]
        price_30d_ago = recent_30["close"].iloc[0]
        ma_7 = recent_30["close"].tail(7).mean()
        ma_14 = recent_30["close"].tail(14).mean()
        ma_30 = recent_30["close"].mean()
        out.append({"symbol": symbol,
                    "return_7d": (current_price - price_7d_ago) / price_7d_ago * 100,
                    "return_14d": (current_price - price_14d_ago) / price_14d_ago * 100,
                    "return_30d": (current_price - price_30d_ago) / price_30d_ago * 100,
                    "ma_7": ma_7, "ma_14": ma_14, "ma_30": ma_30,
                    "trend_strength": (ma_7 - ma_30) / ma_30 * 100,
                    "above_ma7": current_price > ma_7, "above_ma14": current_price > ma_14,
                    "above_ma30": current_price > ma_30})
    return pd.DataFrame(out).set_index("symbol")


def test_trend_metrics_match_the_production_loop():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2026-05-01", periods=45)
    rows = []
    for s, n in (("AAA", 45), ("BBB", 45), ("CCC", 31), ("SHORT", 20)):
        px = 100 * np.cumprod(1 + rng.normal(0, 0.02, size=n))
        rows += [{"date": d, "symbol": s, "close": p} for d, p in zip(dates[-n:], px)]
    df = pd.DataFrame(rows)
    want = _production_trend_loop(df)
    got = sel.trend_metrics(df).set_index("symbol")
    assert "SHORT" not in want.index and np.isnan(got.loc["SHORT", "trend_strength"])
    for s in want.index:
        for c in ["return_7d", "return_14d", "return_30d", "ma_7", "ma_14", "ma_30", "trend_strength"]:
            assert got.loc[s, c] == pytest.approx(want.loc[s, c]), (s, c)
        for c in ["above_ma7", "above_ma14", "above_ma30"]:
            assert bool(got.loc[s, c]) == bool(want.loc[s, c])
    # the panel version agrees with the as-of version on an earlier date
    as_of = dates[-8]
    early = sel.trend_metrics(df, as_of=as_of).set_index("symbol")
    panel = sel.trend_metrics_panel(df)
    row = panel[(panel["date"] == as_of) & (panel["symbol"] == "AAA")].iloc[0]
    assert row["trend_strength"] == pytest.approx(early.loc["AAA", "trend_strength"])
    # the short symbol fails the filter, exactly as its NaN metrics did in production
    mask = sel.trend_mask(got.reset_index(), P)
    assert not mask[got.reset_index()["symbol"] == "SHORT"].iloc[0]


def test_trend_mask_thresholds_and_switch():
    m = pd.DataFrame({"return_7d": [-2.0, -4.0, 0.0, np.nan], "return_14d": [-5.0, 0.0, -7.0, 0.0],
                      "return_30d": [1.0, 1.0, 1.0, 1.0], "trend_strength": [0.0, 0.0, 0.0, 0.0],
                      "above_ma7": [True, True, True, True], "above_ma14": [False, True, True, True]})
    assert sel.trend_mask(m, P).tolist() == [True, False, False, False]
    assert sel.trend_mask(m, sel.SelectionParams(trend_filter=False)).tolist() == [True] * 4
    assert sel.trend_mask(m, sel.SelectionParams(require_above_mas=True)).tolist() == [False, False, False, False]
    assert sel.trend_mask(m, sel.SelectionParams(filter_7d_min=None, filter_14d_min=None)).tolist() == [True, True, True, True]


# 4 + 5. selection and sizing -------------------------------------------------
def test_select_book_top_n_min_confidence_sizing_and_no_trade():
    c = pd.DataFrame({"symbol": list("ABCDEF"), "prediction": [0.05, -0.04, 0.03, 0.001, -0.02, 0.0],
                      "volatility_20d": [0.02, 0.01, np.nan, 0.03, 0.004, 0.02]})
    book = sel.select_book(c, sel.SelectionParams(top_n=4, min_confidence=0.0015))
    assert book["symbol"].tolist() == ["A", "B", "C", "E"]              # top 4 by |pred|, D below min_confidence...
    assert book["side"].tolist() == ["long", "short", "long", "short"]
    # sizing: conf / clip(vol, 0.006); C's NaN vol -> median of the clipped vols
    vols = pd.Series([0.02, 0.01, np.nan, 0.006])
    w = pd.Series([0.05, 0.04, 0.03, 0.02]) / vols.fillna(vols.median())
    np.testing.assert_allclose(book["position_pct"].to_numpy(), (w / w.sum() * 100).to_numpy())
    assert book["position_pct"].sum() == pytest.approx(100.0) and book["rank"].tolist() == [1, 2, 3, 4]
    # fewer than min_stocks -> empty book (no trade)
    assert sel.select_book(c.head(2), sel.SelectionParams(min_stocks=3)).empty
    # inverse-vol off: signal-proportional
    flat = sel.select_book(c, sel.SelectionParams(top_n=3, inv_vol=False))
    np.testing.assert_allclose(flat["position_pct"].to_numpy(), np.array([0.05, 0.04, 0.03]) / 0.12 * 100)


def test_daily_book_end_to_end_and_params_record():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2026-06-01", periods=40)
    syms = [f"S{i:02d}" for i in range(30)]
    prices = pd.DataFrame([{"date": d, "symbol": s, "close": 100 + i * 0.1 + rng.normal()} for i, d in enumerate(dates) for s in syms])
    scores = pd.DataFrame([{"date": d, "symbol": s, "factor_a": rng.normal(), "factor_b": rng.normal()}
                           for d in dates[-5:] for s in syms])
    extra = pd.DataFrame({"symbol": syms, "volatility_20d": rng.uniform(0.01, 0.03, len(syms))})
    book, cands = sel.daily_book(scores, prices, {"a": 0.5, "b": 0.5}, sel.SelectionParams.production(), extra=extra)
    assert 3 <= len(book) <= 10 and set(book["symbol"]) <= set(cands.loc[cands["trend_ok"], "symbol"])
    assert book["position_pct"].sum() == pytest.approx(100.0)
    assert {"prediction", "trend_ok", "return_7d", "volatility_20d"} <= set(cands.columns)
    d = sel.SelectionParams.production().to_dict()
    assert d["version"] == sel.VERSION and d["day_weights"] == [0.30, 0.25, 0.20, 0.15, 0.10] and d["filter_7d_min"] == -3.0
