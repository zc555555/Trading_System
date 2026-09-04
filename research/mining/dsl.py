"""Causal-by-construction factor expression DSL (mining step 2).

The mining agent never writes Python. It emits a one-line expression over a
fixed operator registry, e.g.

    rank(ts_corr(close, volume, 20)) - rank(ts_std(returns, 20))

and this module parses, validates and compiles it into a feature column on
the (date, symbol) panel. Causality is closed under composition:

  * every time-series operator reads a trailing window that ends at the
    current row of the same symbol (rolling / ewm / positive-offset shift);
  * every cross-sectional operator reads only the current date;
  * element-wise operators read only the current row.

A full-day normaliser, a forward shift, or any access to another row's
future is not expressible. The AQuA lesson (Appendix B): the guarantee moves
from "the reviewer should catch leakage" to "leakage cannot be written".

Limits (pre-registered in RULEBOOK.md): windows 1..250, total lookback
<= 250 (the production warm-up), <= 40 nodes, depth <= 8.

The parser accepts a strict subset of Python expression syntax (names,
numeric constants, calls to registered operators, + - * / ** and < >),
walked through `ast`; attribute access, subscripts, lambdas, keywords and
anything else are rejected before evaluation.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_FIELDS = ("open", "high", "low", "close", "volume", "returns", "dollar_volume", "typical")
# auxiliary fields: present only when the panel carries the column (see
# mining/aux_fields.py); an expression using an absent one is refused
AUX_FIELDS = ("marketcap", "turnover", "filing_days", "news_tone", "news_articles",
              "insider_buys", "insider_sells", "insider_net_frac", "short_ratio", "days_to_cover",
              "book_to_market", "earnings_yield", "sales_to_price", "gross_profitability", "roe",
              "asset_growth", "accruals", "leverage", "cash_to_assets", "rd_to_sales",
              "capex_to_assets", "op_margin", "short_vol_ratio", "inst_own", "inst_holders", "inst_top5",
              "wiki_views")
FIELDS = BASE_FIELDS + AUX_FIELDS
FIELD_DOC = {
    "open": "session open", "high": "session high", "low": "session low",
    "close": "session close", "volume": "session share volume",
    "returns": "log(close / close of the previous session), same symbol",
    "dollar_volume": "close * volume",
    "typical": "(high + low + close) / 3",
    "marketcap": "close * latest usable cover-page share count (SEC EDGAR, point-in-time), USD millions",
    "turnover": "volume / latest usable cover-page share count: fraction of shares traded in the session",
    "filing_days": "sessions since the last 10-Q/10-K filing usable today (filing session + 1); NaN before the first filing",
    "news_tone": "GDELT mean news tone (article-weighted, roughly -10..+10) of the calendar days usable at this session (a day's news is usable from the next session); NaN when no article",
    "news_articles": "GDELT article count over the same days; 0 = covered but no article, NaN = symbol not covered",
    "insider_buys": "distinct insiders whose open-market PURCHASE Form 4s became usable at this session (filed day + 1 session); 0 = covered, none; NaN = no SEC mapping",
    "insider_sells": "distinct insiders whose open-market SALE Form 4s became usable at this session; same conventions",
    "insider_net_frac": "(insider shares bought - sold) usable at this session / shares outstanding; NaN when the share count is unknown",
    "short_ratio": "latest PUBLIC FINRA short interest / shares outstanding (semi-monthly, public ~10 sessions after settlement, from 2018); NaN before",
    "days_to_cover": "FINRA short interest / average daily volume for the same figure",
    "book_to_market": "PIT book equity / market cap, from the latest 10-Q/10-K usable today (filing session + 1; XBRL, TTM flows)",
    "earnings_yield": "net income TTM / market cap (PIT)",
    "sales_to_price": "revenue TTM / market cap (PIT)",
    "gross_profitability": "gross profit TTM / total assets (Novy-Marx; PIT)",
    "roe": "net income TTM / book equity (PIT)",
    "asset_growth": "total assets / assets one year earlier - 1 (Cooper-Gulen-Schill; PIT)",
    "accruals": "(net income TTM - operating cash flow TTM) / assets (Sloan; PIT)",
    "leverage": "long-term debt / assets (0 when no long-term debt is reported; PIT)",
    "cash_to_assets": "cash and equivalents / assets (PIT)",
    "rd_to_sales": "R&D TTM / revenue TTM (0 when no R&D is reported; PIT)",
    "capex_to_assets": "capital expenditure TTM / assets (PIT)",
    "op_margin": "operating income TTM / revenue TTM (PIT)",
    "short_vol_ratio": "FINRA Reg SHO short sale volume / total reported volume of the previous session (daily, from 2018); NaN when absent",
    "inst_own": "13F institutional shares / shares outstanding for the latest quarter whose 45-day filing deadline has passed (quarterly steps, carried <= 70 sessions)",
    "inst_holders": "number of 13F managers holding the stock, same timing (Chen-Hong-Stein breadth: use delta over ~63 sessions)",
    "inst_top5": "share of the institutional holdings held by the five largest managers, same timing",
    "wiki_views": "English-Wikipedia page views of the company article over the calendar days usable at this session (a day is usable from the next session; weekend days land on Monday), from 2015-07; NaN when no article or before its first day",
}
MAX_WINDOW = 250
MAX_LOOKBACK = 250
MAX_NODES = 40
MAX_DEPTH = 8

# name -> (kind, signature, doc); signature letters: x = series-valued
# argument, w = integer window constant, c = numeric constant.
OPS: dict[str, tuple[str, tuple[str, ...], str]] = {
    # time-series (per symbol, trailing window ending today)
    "delay":        ("ts", ("x", "w"), "value w sessions ago"),
    "delta":        ("ts", ("x", "w"), "x - delay(x, w)"),
    "pct_change":   ("ts", ("x", "w"), "x / delay(x, w) - 1"),
    "ts_mean":      ("ts", ("x", "w"), "trailing mean over w sessions"),
    "ts_std":       ("ts", ("x", "w"), "trailing sample std over w sessions"),
    "ts_sum":       ("ts", ("x", "w"), "trailing sum over w sessions"),
    "ts_min":       ("ts", ("x", "w"), "trailing min over w sessions"),
    "ts_max":       ("ts", ("x", "w"), "trailing max over w sessions"),
    "ts_rank":      ("ts", ("x", "w"), "percentile rank of today's value within the trailing w sessions"),
    "ts_argmax":    ("ts", ("x", "w"), "sessions since the trailing-w max (0 = today)"),
    "ts_argmin":    ("ts", ("x", "w"), "sessions since the trailing-w min (0 = today)"),
    "ts_zscore":    ("ts", ("x", "w"), "(x - ts_mean) / ts_std over w sessions"),
    "decay_linear": ("ts", ("x", "w"), "linearly-weighted trailing mean (today weight w)"),
    "ema":          ("ts", ("x", "w"), "exponential moving average, span w"),
    "ts_corr":      ("ts", ("x", "x", "w"), "trailing Pearson correlation of two series over w sessions"),
    "ts_cov":       ("ts", ("x", "x", "w"), "trailing covariance of two series over w sessions"),
    # cross-sectional (per date, across symbols)
    "rank":         ("cs", ("x",), "percentile rank across symbols on the same date"),
    "zscore":       ("cs", ("x",), "z-score across symbols on the same date"),
    "sector_rank":  ("cs", ("x",), "percentile rank across symbols of the SAME SECTOR on the same date (sector-neutral)"),
    "sector_demean": ("cs", ("x",), "x minus the same-date mean of its sector (sector-neutral level)"),
    "demean":       ("cs", ("x",), "x minus the same-date cross-sectional mean"),
    # element-wise
    "add":   ("ew", ("x", "x"), "a + b"),
    "sub":   ("ew", ("x", "x"), "a - b"),
    "mul":   ("ew", ("x", "x"), "a * b"),
    "div":   ("ew", ("x", "x"), "a / b (NaN where b == 0)"),
    "max":   ("ew", ("x", "x"), "element-wise max"),
    "min":   ("ew", ("x", "x"), "element-wise min"),
    "gt":    ("ew", ("x", "x"), "1.0 where a > b else 0.0"),
    "lt":    ("ew", ("x", "x"), "1.0 where a < b else 0.0"),
    "where": ("ew", ("x", "x", "x"), "b where cond > 0 else c"),
    "neg":   ("ew", ("x",), "-x"),
    "abs":   ("ew", ("x",), "|x|"),
    "sign":  ("ew", ("x",), "sign(x)"),
    "log":   ("ew", ("x",), "sign(x) * log(1 + |x|)  (symmetric, defined everywhere)"),
    "sqrt":  ("ew", ("x",), "sign(x) * sqrt(|x|)"),
    "pow":   ("ew", ("x", "c"), "sign(x) * |x| ** c"),
    "clip":  ("ew", ("x", "c", "c"), "clip x to [lo, hi]"),
    "fillna": ("ew", ("x", "c"), "x with missing values replaced by c, but only from the symbol's first observation on (a source that starts in 2018 stays NaN before 2018)"),
}
_BINOPS: dict[type, str] = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_CMPOPS: dict[type, str] = {ast.Gt: "gt", ast.Lt: "lt"}


class DSLError(ValueError):
    """Raised for any expression the registry does not admit."""


@dataclass(frozen=True)
class Const:
    value: float

    def __str__(self) -> str:
        v = self.value
        return str(int(v)) if float(v).is_integer() else repr(float(v))


@dataclass(frozen=True)
class Node:
    op: str                     # operator name, or "field"
    args: tuple                 # Node | Const, or (field_name,) for "field"

    def __str__(self) -> str:
        if self.op == "field":
            return self.args[0]
        return f"{self.op}({', '.join(str(a) for a in self.args)})"


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------
def parse(expr: str) -> Node:
    """Parse a DSL expression string into a Node tree (no evaluation)."""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error: {e.msg}") from None
    node = _convert(tree.body)
    if isinstance(node, Const):
        raise DSLError("expression is a constant; it must reference at least one field")
    return node


def _convert(n: ast.AST):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
            raise DSLError(f"unsupported constant {n.value!r}")
        return Const(float(n.value))
    if isinstance(n, ast.Name):
        if n.id in FIELDS:
            return Node("field", (n.id,))
        if n.id in OPS:
            raise DSLError(f"operator '{n.id}' used without arguments")
        raise DSLError(f"unknown name '{n.id}' (fields: {', '.join(FIELDS)})")
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        child = _convert(n.operand)
        return Const(-child.value) if isinstance(child, Const) else Node("neg", (child,))
    if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
        op = _BINOPS[type(n.op)]
        left, right = _convert(n.left), _convert(n.right)
        if isinstance(left, Const) and isinstance(right, Const):
            raise DSLError("constant-only arithmetic: fold it yourself")
        if op == "pow" and not isinstance(right, Const):
            raise DSLError("pow exponent must be a numeric constant")
        return _make(op, (left, right))
    if isinstance(n, ast.Compare):
        if len(n.ops) != 1 or type(n.ops[0]) not in _CMPOPS:
            raise DSLError("only a single '<' or '>' comparison is allowed")
        return _make(_CMPOPS[type(n.ops[0])], (_convert(n.left), _convert(n.comparators[0])))
    if isinstance(n, ast.Call):
        if not isinstance(n.func, ast.Name) or n.func.id not in OPS:
            raise DSLError("calls must be to a registered operator by bare name")
        if n.keywords:
            raise DSLError("keyword arguments are not allowed")
        return _make(n.func.id, tuple(_convert(a) for a in n.args))
    raise DSLError(f"unsupported syntax: {type(n).__name__}")


def _make(op: str, args: tuple) -> Node:
    kind, sig, _ = OPS[op]
    if len(args) != len(sig):
        raise DSLError(f"{op} expects {len(sig)} argument(s), got {len(args)}")
    for a, s in zip(args, sig):
        if s == "x":
            continue
        if not isinstance(a, Const):
            raise DSLError(f"{op}: argument must be a numeric constant")
        if s == "w":
            v = a.value
            lo = 1 if op == "delay" else 2
            if not float(v).is_integer() or not (lo <= int(v) <= MAX_WINDOW):
                raise DSLError(f"{op}: window must be an integer in [{lo}, {MAX_WINDOW}], got {v}")
    if kind == "ts" and all(isinstance(a, Const) for a in args[:-1]):
        raise DSLError(f"{op}: needs a series argument, not only constants")
    if op == "clip" and args[1].value > args[2].value:
        raise DSLError("clip: lo must be <= hi")
    return Node(op, args)


# --------------------------------------------------------------------------
# validate / describe
# --------------------------------------------------------------------------
def count_nodes(node) -> int:
    if isinstance(node, Const):
        return 0
    if node.op == "field":
        return 1
    return 1 + sum(count_nodes(a) for a in node.args)


def depth(node) -> int:
    if isinstance(node, Const):
        return 0
    if node.op == "field":
        return 1
    return 1 + max(depth(a) for a in node.args)


def lookback(node) -> int:
    """Conservative number of prior sessions the expression needs (nested
    windows add). `returns` needs one prior session."""
    if isinstance(node, Const):
        return 0
    if node.op == "field":
        return 1 if node.args[0] == "returns" else 0
    kind, sig, _ = OPS[node.op]
    child = max(lookback(a) for a in node.args)
    if kind == "ts":
        w = int(node.args[-1].value)
        return child + w
    return child


def fields_used(node) -> set[str]:
    if isinstance(node, Const):
        return set()
    if node.op == "field":
        return {node.args[0]}
    out: set[str] = set()
    for a in node.args:
        out |= fields_used(a)
    return out


def validate(node: Node) -> dict:
    """Enforce size limits; returns the expression's stats."""
    n, d, lb = count_nodes(node), depth(node), lookback(node)
    if n > MAX_NODES:
        raise DSLError(f"expression has {n} nodes > {MAX_NODES}")
    if d > MAX_DEPTH:
        raise DSLError(f"expression depth {d} > {MAX_DEPTH}")
    if lb > MAX_LOOKBACK:
        raise DSLError(f"expression lookback {lb} sessions > {MAX_LOOKBACK} (production warm-up)")
    return {"n_nodes": n, "depth": d, "lookback": lb, "canonical": str(node)}


def canonical(expr: str) -> str:
    return str(parse(expr))


def expression_hash(expr: str) -> str:
    return hashlib.sha1(canonical(expr).encode("utf-8")).hexdigest()[:12]


def describe_ops() -> str:
    lines = ["FIELDS (price-volume)"]
    lines += [f"  {f:<14}{FIELD_DOC[f]}" for f in BASE_FIELDS]
    lines.append("FIELDS (auxiliary, Sharadar; same causality rules)")
    lines += [f"  {f:<14}{FIELD_DOC[f]}" for f in AUX_FIELDS]
    for kind, title in (("ts", "TIME-SERIES (per symbol, trailing window ending today)"),
                        ("cs", "CROSS-SECTIONAL (per date, across symbols)"),
                        ("ew", "ELEMENT-WISE")):
        lines.append(title)
        for name, (k, sig, doc) in OPS.items():
            if k == kind:
                argtxt = ", ".join({"x": "x", "w": "w:int", "c": "c:num"}[s] for s in sig)
                lines.append(f"  {name}({argtxt})".ljust(28) + doc)
    lines.append("SYNTAX  + - * / ** < > and unary minus map to add/sub/mul/div/pow/lt/gt/neg.")
    lines.append(f"LIMITS  windows 1..{MAX_WINDOW}; total lookback <= {MAX_LOOKBACK}; "
                 f"<= {MAX_NODES} nodes; depth <= {MAX_DEPTH}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# compile / evaluate
# --------------------------------------------------------------------------
class _Ctx:
    def __init__(self, frame: pd.DataFrame, sym: pd.Series, dt: pd.Series):
        self.f = frame
        self.sym = sym
        self.dt = dt
        self.sector = frame["sector"] if "sector" in frame.columns else None
        self._fields: dict[str, pd.Series] = {}

    def field(self, name: str) -> pd.Series:
        if name in self._fields:
            return self._fields[name]
        f = self.f
        if name in ("open", "high", "low", "close", "volume"):
            s = f[name].astype(float)
        elif name == "returns":
            s = np.log(f["close"].astype(float)).groupby(self.sym, sort=False).diff(1)
        elif name == "dollar_volume":
            s = f["close"].astype(float) * f["volume"].astype(float)
        elif name == "typical":
            s = (f["high"].astype(float) + f["low"].astype(float) + f["close"].astype(float)) / 3.0
        elif name in AUX_FIELDS:
            s = f[name].astype(float)
        else:  # pragma: no cover
            raise DSLError(f"unknown field {name}")
        self._fields[name] = s
        return s

    # -- per-symbol trailing windows ------------------------------------
    def roll(self, s: pd.Series, w: int, fn):
        return s.groupby(self.sym, sort=False).transform(
            lambda g: fn(g.rolling(w, min_periods=w)))

    def shift(self, s: pd.Series, w: int) -> pd.Series:
        assert w >= 1
        return s.groupby(self.sym, sort=False).shift(w)


def _series(ctx: _Ctx, v):
    """Broadcast a Const to a full-length Series."""
    if isinstance(v, pd.Series):
        return v
    return pd.Series(float(v), index=ctx.f.index)


def _eval(node, ctx: _Ctx):
    if isinstance(node, Const):
        return float(node.value)
    if node.op == "field":
        return ctx.field(node.args[0])

    op = node.op
    kind, sig, _ = OPS[op]
    if kind == "ts":
        w = int(node.args[-1].value)
        xs = [_series(ctx, _eval(a, ctx)) for a in node.args[:-1]]
        x = xs[0]
        if op == "delay":
            return ctx.shift(x, w)
        if op == "delta":
            return x - ctx.shift(x, w)
        if op == "pct_change":
            prev = ctx.shift(x, w)
            return x / prev.where(prev != 0) - 1.0
        if op == "ts_mean":
            return ctx.roll(x, w, lambda r: r.mean())
        if op == "ts_std":
            return ctx.roll(x, w, lambda r: r.std())
        if op == "ts_sum":
            return ctx.roll(x, w, lambda r: r.sum())
        if op == "ts_min":
            return ctx.roll(x, w, lambda r: r.min())
        if op == "ts_max":
            return ctx.roll(x, w, lambda r: r.max())
        if op == "ts_rank":
            return ctx.roll(x, w, lambda r: r.rank(pct=True))
        if op == "ts_argmax":
            return ctx.roll(x, w, lambda r: r.apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True))
        if op == "ts_argmin":
            return ctx.roll(x, w, lambda r: r.apply(lambda a: len(a) - 1 - int(np.argmin(a)), raw=True))
        if op == "ts_zscore":
            m = ctx.roll(x, w, lambda r: r.mean())
            sd = ctx.roll(x, w, lambda r: r.std())
            return (x - m) / sd.where(sd != 0)
        if op == "decay_linear":
            wts = np.arange(1, w + 1, dtype=float)
            return ctx.roll(x, w, lambda r: r.apply(lambda a: float(np.dot(a, wts)) / wts.sum(), raw=True))
        if op == "ema":
            return x.groupby(ctx.sym, sort=False).transform(
                lambda g: g.ewm(span=w, min_periods=w).mean())
        if op in ("ts_corr", "ts_cov"):
            y = xs[1]
            pair = pd.DataFrame({"x": x, "y": y})
            if op == "ts_corr":
                fn = lambda g: g["x"].rolling(w, min_periods=w).corr(g["y"])  # noqa: E731
            else:
                fn = lambda g: g["x"].rolling(w, min_periods=w).cov(g["y"])   # noqa: E731
            return pair.groupby(ctx.sym, sort=False, group_keys=False).apply(fn)
        raise DSLError(f"unhandled ts op {op}")  # pragma: no cover

    if kind == "cs":
        x = _series(ctx, _eval(node.args[0], ctx))
        if op in ("sector_rank", "sector_demean"):
            if ctx.sector is None:
                raise DSLError("sector-neutral operators need a `sector` column on the panel (aux_fields.attach_sector)")
            gs = x.groupby([ctx.dt, ctx.sector], sort=False)
            return gs.rank(pct=True) if op == "sector_rank" else x - gs.transform("mean")
        g = x.groupby(ctx.dt, sort=False)
        if op == "rank":
            return g.rank(pct=True)
        if op == "zscore":
            sd = g.transform("std")
            return (x - g.transform("mean")) / sd.where(sd != 0)
        if op == "demean":
            return x - g.transform("mean")
        raise DSLError(f"unhandled cs op {op}")  # pragma: no cover

    # element-wise
    vals = [_eval(a, ctx) for a in node.args]
    if op == "add":
        return vals[0] + vals[1]
    if op == "sub":
        return vals[0] - vals[1]
    if op == "mul":
        return vals[0] * vals[1]
    if op == "div":
        b = vals[1]
        if isinstance(b, pd.Series):
            return vals[0] / b.where(b != 0)
        return vals[0] / b if b != 0 else _series(ctx, np.nan)
    if op in ("max", "min"):
        a, b = _series(ctx, vals[0]), _series(ctx, vals[1])
        return pd.Series((np.maximum if op == "max" else np.minimum)(a.to_numpy(), b.to_numpy()),
                         index=ctx.f.index)
    if op in ("gt", "lt"):
        a, b = _series(ctx, vals[0]), _series(ctx, vals[1])
        out = (a > b) if op == "gt" else (a < b)
        return out.astype(float).where(a.notna() & b.notna())
    if op == "where":
        c, a, b = (_series(ctx, v) for v in vals)
        return pd.Series(np.where(c.to_numpy() > 0, a.to_numpy(), b.to_numpy()),
                         index=ctx.f.index).where(c.notna())
    x = vals[0]
    if op == "neg":
        return -x
    if op == "abs":
        return np.abs(x)
    if op == "sign":
        return np.sign(x)
    if op == "log":
        return np.sign(x) * np.log1p(np.abs(x))
    if op == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    if op == "pow":
        return np.sign(x) * np.abs(x) ** float(vals[1])
    if op == "fillna":
        xv = x.to_numpy(dtype=float)
        # leading NaNs (before the symbol's first valid observation) are "not covered yet", not "missing":
        # filling them would fabricate a constant history before the source starts
        seen = pd.Series(~np.isnan(xv), index=x.index).groupby(ctx.f["symbol"].to_numpy()).cummax().to_numpy()
        return pd.Series(np.where(np.isnan(xv) & seen, float(vals[1]), xv), index=x.index)
    if op == "clip":
        return np.clip(x, float(vals[1]), float(vals[2]))
    raise DSLError(f"unhandled op {op}")  # pragma: no cover


def compile_expression(expr: str, df: pd.DataFrame,
                       date_col: str = "date", symbol_col: str = "symbol") -> pd.Series:
    """Evaluate `expr` on a long panel with columns [date, symbol, open, high,
    low, close, volume]. Returns a float Series aligned to `df.index`, in
    `df`'s row order (the panel may be unsorted). Inf becomes NaN."""
    node = parse(expr)
    validate(node)
    aux_used = sorted(f for f in fields_used(node) if f in AUX_FIELDS)
    absent = [f for f in aux_used if f not in df.columns]
    if absent:
        raise DSLError(f"field(s) {absent} not available in this panel (source data not attached)")
    need = [date_col, symbol_col, "open", "high", "low", "close", "volume"] + aux_used
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DSLError(f"panel is missing columns {missing}")
    if "sector" in df.columns:                        # categorical label for the sector-neutral operators
        need.append("sector")

    sym_codes, _ = pd.factorize(df[symbol_col], sort=True)
    dt_key = pd.to_datetime(df[date_col]).astype("int64").to_numpy()
    order = np.lexsort((dt_key, sym_codes))          # by symbol, then date
    frame = df.iloc[order][need].reset_index(drop=True)
    ctx = _Ctx(frame, frame[symbol_col], frame[date_col])

    out = _eval(node, ctx)
    out = _series(ctx, out).astype(float).replace([np.inf, -np.inf], np.nan)
    result = np.full(len(df), np.nan)
    result[order] = out.to_numpy()
    return pd.Series(result, index=df.index, name=canonical(expr))
