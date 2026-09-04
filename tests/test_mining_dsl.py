"""Pins the mining DSL (research/mining/dsl.py): the parser admits only the
registry, size limits hold, operators reproduce hand arithmetic, row order
does not matter, and -- the point of the module -- no expression can be
influenced by future rows (metamorphic check over every operator plus
random compositions)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import dsl  # noqa: E402


def _panel(n_syms=4, n_days=300, seed=0, tz="America/New_York"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, tz=tz)
    rows = []
    for i in range(n_syms):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        hi = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        lo = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i}", "open": close * (1 + rng.normal(0, 0.002, n_days)),
            "high": hi, "low": lo, "close": close,
            "volume": np.exp(rng.normal(14, 0.3, n_days)),
            "sector": "A" if i % 2 == 0 else "B",          # for the sector-neutral operators
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


ALL_OPS_EXPRS = [
    "delay(close, 3)", "delta(close, 5)", "pct_change(close, 10)", "ts_mean(returns, 5)",
    "ts_std(returns, 20)", "ts_sum(volume, 5)", "ts_min(low, 10)", "ts_max(high, 10)",
    "ts_rank(close, 20)", "ts_argmax(high, 15)", "ts_argmin(low, 15)", "ts_zscore(close, 20)",
    "decay_linear(returns, 10)", "ema(close, 12)", "ts_corr(close, volume, 20)",
    "ts_cov(returns, dollar_volume, 20)", "rank(close)", "zscore(typical)", "demean(returns)",
    "add(close, open)", "sub(high, low)", "mul(close, volume)", "div(close, open)",
    "max(open, close)", "min(open, close)", "gt(close, open)", "lt(close, open)",
    "where(gt(close, open), volume, neg(volume))", "neg(returns)", "abs(returns)",
    "sign(returns)", "log(volume)", "sqrt(dollar_volume)", "pow(returns, 2)",
    "clip(returns, -0.02, 0.02)",
    "rank(ts_corr(close, volume, 20)) - rank(ts_std(returns, 20))",
    "(close - ts_mean(close, 20)) / ts_std(close, 20)",
    "-delta(close, 5) / close", "close ** 0.5", "close > delay(close, 1)",
]


# ---- parse / validate ------------------------------------------------------
def test_parse_canonical_form():
    assert dsl.canonical("rank(ts_corr(close, volume, 20)) - rank(ts_std(returns,20))") == \
        "sub(rank(ts_corr(close, volume, 20)), rank(ts_std(returns, 20)))"
    assert dsl.canonical("-close") == "neg(close)"
    assert dsl.canonical("close ** 2") == "pow(close, 2)"
    assert dsl.canonical("close > open") == "gt(close, open)"
    assert dsl.expression_hash("close - open") == dsl.expression_hash("sub(close,  open)")


@pytest.mark.parametrize("bad", [
    "close.shift(-1)", "__import__('os')", "close[0]", "lambda: close", "foo(close)",
    "ts_mean(close, 300)", "ts_mean(close, 0)", "ts_mean(close, 2.5)", "delay(close, -1)",
    "delay(close, 0)", "2 + 3", "rank(close, close)", "ts_mean(5, 20)", "ts_mean(close, w=20)",
    "close ** open", "close >= open", "close > open > low", "42", "", "clip(close, 1, 0)",
    "rank", "close if close else open", "[close]", "close.T",
])
def test_rejects_anything_outside_the_registry(bad):
    with pytest.raises(dsl.DSLError):
        dsl.parse(bad)


def test_limits_nodes_depth_lookback():
    assert dsl.lookback(dsl.parse("ts_mean(ts_std(returns, 20), 10)")) == 31
    assert dsl.lookback(dsl.parse("delay(close, 250)")) == 250
    with pytest.raises(dsl.DSLError, match="lookback"):
        dsl.validate(dsl.parse("ts_mean(delay(close, 200), 100)"))
    deep = "close"
    for _ in range(9):
        deep = f"abs({deep})"
    with pytest.raises(dsl.DSLError, match="depth"):
        dsl.validate(dsl.parse(deep))
    wide = " + ".join(["close"] * 25)
    with pytest.raises(dsl.DSLError, match="nodes"):
        dsl.validate(dsl.parse(wide))


def test_describe_ops_lists_every_operator_and_field():
    doc = dsl.describe_ops()
    for name in dsl.OPS:
        assert f"{name}(" in doc
    for f in dsl.FIELDS:
        assert f in doc
    assert "auxiliary" in doc


# ---- values ----------------------------------------------------------------
def test_operators_reproduce_hand_arithmetic():
    df = _panel()
    g = df.groupby("symbol")["close"]
    pd.testing.assert_series_equal(
        dsl.compile_expression("delay(close, 1)", df), g.shift(1).astype(float),
        check_names=False)
    pd.testing.assert_series_equal(
        dsl.compile_expression("ts_mean(close, 3)", df),
        g.transform(lambda s: s.rolling(3, min_periods=3).mean()), check_names=False)
    pd.testing.assert_series_equal(
        dsl.compile_expression("delta(close, 2)", df), g.diff(2), check_names=False)
    pd.testing.assert_series_equal(
        dsl.compile_expression("rank(close)", df), df.groupby("date")["close"].rank(pct=True),
        check_names=False)
    ret = np.log(df["close"]).groupby(df["symbol"]).diff(1)
    pd.testing.assert_series_equal(dsl.compile_expression("returns", df), ret, check_names=False)
    z = dsl.compile_expression("zscore(close)", df)
    assert abs(z.groupby(df["date"]).mean().abs().max()) < 1e-9
    argmax = dsl.compile_expression("ts_argmax(close, 5)", df).dropna()
    assert argmax.between(0, 4).all()
    d = dsl.compile_expression("div(close, sub(close, close))", df)
    assert d.isna().all(), "division by zero must be NaN, never inf"


def test_row_order_does_not_matter():
    df = _panel()
    expr = "rank(ts_corr(close, volume, 10)) - ts_zscore(returns, 15)"
    a = dsl.compile_expression(expr, df)
    shuffled = df.sample(frac=1.0, random_state=1)
    b = dsl.compile_expression(expr, shuffled).reindex(df.index)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_all_ops_compile_on_a_panel():
    df = _panel()
    for expr in ALL_OPS_EXPRS:
        out = dsl.compile_expression(expr, df)
        assert len(out) == len(df) and out.index.equals(df.index), expr
        assert np.isfinite(out.dropna()).all(), expr
        assert out.notna().sum() > 0, expr


# ---- the point: no future influence ---------------------------------------
def _rand_expr(rng, d=0):
    if d >= 3 or rng.random() < 0.3:
        return str(rng.choice(dsl.BASE_FIELDS))
    op = str(rng.choice(list(dsl.OPS)))
    _, sig, _ = dsl.OPS[op]
    args = []
    for s in sig:
        if s == "x":
            args.append(_rand_expr(rng, d + 1))
        elif s == "w":
            args.append(str(int(rng.integers(2, 25))))
        else:
            args.append(str(round(float(rng.uniform(0.5, 2.0)), 2)))
    if op == "clip":
        args[1], args[2] = "-1", "1"
    return f"{op}({', '.join(args)})"


def _perturb_future(df, cutoff_idx, rng):
    out = df.copy()
    dates = np.sort(df["date"].unique())
    fut = out["date"] >= dates[cutoff_idx]
    for c in ("open", "high", "low", "close", "volume"):
        out.loc[fut, c] = out.loc[fut, c] * rng.uniform(0.5, 1.5, fut.sum())
    return out, fut


def test_future_rows_never_change_past_values():
    rng = np.random.default_rng(7)
    df = _panel()
    exprs = list(ALL_OPS_EXPRS)
    while len(exprs) < len(ALL_OPS_EXPRS) + 25:
        e = _rand_expr(rng)
        try:
            dsl.validate(dsl.parse(e))
        except dsl.DSLError:
            continue
        exprs.append(e)
    mutated, fut = _perturb_future(df, 260, rng)
    past = ~fut.to_numpy()
    for expr in exprs:
        a = dsl.compile_expression(expr, df).to_numpy()[past]
        b = dsl.compile_expression(expr, mutated).to_numpy()[past]
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg=expr)


def test_cross_sectional_ops_read_only_the_same_date():
    rng = np.random.default_rng(3)
    df = _panel()
    dates = np.sort(df["date"].unique())
    t = dates[150]
    mutated = df.copy()
    hit = (mutated["date"] == t) & (mutated["symbol"] == "S0")
    mutated.loc[hit, "close"] *= 3.0
    for expr in ("rank(close)", "zscore(close)", "demean(close)"):
        a = dsl.compile_expression(expr, df)
        b = dsl.compile_expression(expr, mutated)
        other_dates = (df["date"] != t).to_numpy()
        np.testing.assert_allclose(a.to_numpy()[other_dates], b.to_numpy()[other_dates], equal_nan=True)
    # a time-series op on S0 must not touch any other symbol at all
    a = dsl.compile_expression("ts_mean(close, 5)", df)
    b = dsl.compile_expression("ts_mean(close, 5)", mutated)
    others = (df["symbol"] != "S0").to_numpy()
    np.testing.assert_allclose(a.to_numpy()[others], b.to_numpy()[others], equal_nan=True)
