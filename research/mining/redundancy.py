"""Screen-stage redundancy clustering (screen rule refinement, 2026-09-02).

Eleven passes that are all "how spiky was this stock's volume" are one
idea, not eleven. Before any candidate is promoted to the 35-minute full
stage, every screen pass is compared, by mean per-date cross-sectional
Spearman correlation on the development segment, with

  * a fixed list of incumbent production features (INCUMBENT_REFS),
  * every earlier screen pass that is a cluster representative (any run),
  * the other passes in the same batch.

Pairs at |corr| >= CORR_THRESHOLD are joined into a cluster (union-find).
One representative per cluster: an earlier run's representative keeps
precedence; otherwise the highest |dev t| in the batch. A pass whose best
match is an incumbent feature is never a representative (it is a restatement
of production). Non-representatives carry `redundant_with`; only
representatives may enter the full stage (harness `full` refuses the rest
without --force).

`residual_dev_t` is informational: the candidate's per-date ranks are
orthogonalised against its single most-correlated reference (cross-sectional
OLS per date) and the residual's rank IC against the label is re-tested with
Newey-West (19 lags). It answers "how much is left once the look-alike is
removed" and is returned to the agent so it spends budget elsewhere.

This is AlphaCrafter's semantic de-duplication done with data instead of an
LLM judgement: measurable, reproducible, and it never touches the hidden
tiers (development rows only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent
RESULTS = RESEARCH / "evaluation" / "results"
SURV_PANEL = RESEARCH / "data" / "stocks_with_time_windows_surv.parquet"
REF_CACHE = RESULTS / "_mining_incumbent_ref.parquet"
PASS_CACHE = RESULTS / "_mining_pass_features.parquet"
PASS_CACHE_META = RESULTS / "_mining_pass_features.json"

INCUMBENT_REFS = [
    "volume_std_20d", "volume_ratio_20d", "volume_ratio_5d", "vwap_ratio", "mfi_14",
    "volatility_20d", "volatility_60d", "bb_width",
    "returns_20d", "returns_60d", "momentum_50d", "rsi_7", "macd_hist", "cci_20",
    "pct_52w_high", "kc_position",
]
CORR_THRESHOLD = 0.7
DATE_STRIDE = 5          # sample every 5th dev date for the pairwise correlations
MIN_NAMES = 40
NW_LAGS = 19
RESIDUAL_MIN_CORR = 0.3  # below this the "most correlated reference" is noise; residual t = own t


# --------------------------------------------------------------------------
# pure statistics
# --------------------------------------------------------------------------
def mean_rank_corr(dates: np.ndarray, a: np.ndarray, b: np.ndarray,
                   stride: int = DATE_STRIDE, min_names: int = MIN_NAMES) -> float:
    """Mean over (sampled) dates of the cross-sectional Spearman correlation."""
    df = pd.DataFrame({"date": dates, "a": a, "b": b}).dropna()
    if df.empty:
        return 0.0
    keep = np.sort(df["date"].unique())[::max(int(stride), 1)]
    df = df[df["date"].isin(keep)]
    vals = []
    for _, g in df.groupby("date", sort=False):
        if len(g) < min_names:
            continue
        ra, rb_ = g["a"].rank().to_numpy(), g["b"].rank().to_numpy()
        if ra.std() == 0 or rb_.std() == 0:
            continue
        vals.append(float(np.corrcoef(ra, rb_)[0, 1]))
    return float(np.mean(vals)) if vals else 0.0


def residual_rank_ic(dates: np.ndarray, cand: np.ndarray, ref: np.ndarray, label: np.ndarray,
                     min_names: int = MIN_NAMES) -> pd.Series:
    """Per-date Spearman IC of the candidate's ranks orthogonalised (per date,
    OLS) against the reference's ranks."""
    df = pd.DataFrame({"date": dates, "c": cand, "r": ref, "y": label}).dropna()
    out = {}
    for d, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        rc, rr, ry = g["c"].rank().to_numpy(), g["r"].rank().to_numpy(), g["y"].rank().to_numpy()
        rr_c = rr - rr.mean()
        var = float((rr_c ** 2).sum())
        beta = float((rc - rc.mean()) @ rr_c) / var if var > 0 else 0.0
        res = rc - beta * rr
        if res.std() == 0 or ry.std() == 0:
            continue
        out[d] = float(np.corrcoef(pd.Series(res).rank().to_numpy(), ry)[0, 1])
    return pd.Series(out)


# --------------------------------------------------------------------------
# reference sets
# --------------------------------------------------------------------------
def load_incumbent_refs(panel: pd.DataFrame, source: Path = SURV_PANEL,
                        cache: Path = REF_CACHE) -> pd.DataFrame:
    """Incumbent feature columns aligned to `panel`'s rows (date, symbol)."""
    if cache.exists():
        ref = pd.read_parquet(cache)
        if len(ref) == len(panel) and list(ref["symbol"].iloc[:3]) == list(panel["symbol"].iloc[:3]):
            return ref.drop(columns=["date", "symbol"])
    import pyarrow.parquet as pq
    have = [c for c in INCUMBENT_REFS if c in pq.ParquetFile(source).schema_arrow.names]
    src = pd.read_parquet(source, columns=["date", "symbol"] + have)
    ref = panel[["date", "symbol"]].merge(src, on=["date", "symbol"], how="left")
    ref.to_parquet(cache, index=False)
    return ref.drop(columns=["date", "symbol"])


def prior_representatives(ledger_path: Path, panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compiled features of every earlier screen pass that is a cluster
    representative (or predates clustering), keyed by candidate id."""
    from mining import dsl
    from evaluation import rulebook as rb
    if not Path(ledger_path).exists():
        return {}
    led = rb.load_mined_ledger(ledger_path)
    if led.empty or "screen_pass" not in led.columns:
        return {}
    scr = led[(led["stage"].astype(str) == "screen") & (led["screen_pass"].map(lambda v: v is True))]
    if "cluster_rep" in scr.columns:
        scr = scr[scr["cluster_rep"].isna() | scr["cluster_rep"].map(lambda v: v is True)]
    scr = scr.drop_duplicates("canonical", keep="last")
    if scr.empty:
        return {}

    cache = pd.read_parquet(PASS_CACHE) if PASS_CACHE.exists() else pd.DataFrame()
    meta = json.loads(PASS_CACHE_META.read_text(encoding="utf-8")) if PASS_CACHE_META.exists() else {}
    if len(cache) and len(cache) != len(panel):
        cache, meta = pd.DataFrame(), {}
    out, dirty = {}, False
    for _, r in scr.iterrows():
        canon = str(r["canonical"])
        h = dsl.expression_hash(canon)
        if h not in cache.columns:
            cache[h] = dsl.compile_expression(canon, panel).to_numpy()
            meta[h] = {"candidate_id": str(r["candidate_id"]), "canonical": canon}
            dirty = True
        out[str(r["candidate_id"])] = cache[h].to_numpy()
    if dirty:
        cache.to_parquet(PASS_CACHE, index=False)
        PASS_CACHE_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def assess(passes: list[dict], dates: np.ndarray, label: np.ndarray,
           refs: dict[str, np.ndarray], priors: dict[str, np.ndarray],
           threshold: float = CORR_THRESHOLD) -> dict[str, dict]:
    """`passes`: [{candidate_id, dev_t, x (np.ndarray on the same rows as
    dates/label)}]. `refs`/`priors`: name -> array on the same rows. Returns
    per candidate: max_corr, corr_with, cluster_id, cluster_rep,
    redundant_with, residual_dev_ic, residual_dev_t."""
    ids = [p["candidate_id"] for p in passes]
    x = {p["candidate_id"]: np.asarray(p["x"], dtype=float) for p in passes}
    t = {p["candidate_id"]: float(p["dev_t"]) for p in passes}

    # pairwise correlations inside the batch, computed once, seen from both sides
    pair: dict[tuple[str, str], float] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            c = mean_rank_corr(dates, x[a], x[b])
            pair[(a, b)] = pair[(b, a)] = c

    best: dict[str, tuple[str, float, str]] = {}      # cid -> (name, corr, kind)
    edges: list[tuple[str, str]] = []
    for cid in ids:
        cands = []
        for name, arr in refs.items():
            cands.append(("incumbent", name, mean_rank_corr(dates, x[cid], arr)))
        for name, arr in priors.items():
            cands.append(("prior", name, mean_rank_corr(dates, x[cid], arr)))
        for other in ids:
            if other != cid:
                cands.append(("batch", other, pair[(cid, other)]))
        for kind, name, c in cands:
            if abs(c) >= threshold and kind in ("prior", "batch"):
                edges.append((cid, name))
        if cands:
            kind, name, c = max(cands, key=lambda z: abs(z[2]))
            best[cid] = (name, float(c), kind)
        else:
            best[cid] = ("", 0.0, "")

    # union-find over batch ids + prior ids
    parent = {n: n for n in ids + list(priors)}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        ra, rb_ = find(a), find(b)
        if ra != rb_:
            parent[ra] = rb_
    clusters: dict[str, list[str]] = {}
    for n in ids + list(priors):
        clusters.setdefault(find(n), []).append(n)

    out = {}
    for members in clusters.values():
        prior_members = [m for m in members if m in priors]
        batch_members = [m for m in members if m in x]
        if not batch_members:
            continue
        if prior_members:
            rep = prior_members[0]
        else:
            rep = max(batch_members, key=lambda m: abs(t[m]))
        for m in batch_members:
            name, c, kind = best[m]
            incumbent_hit = kind == "incumbent" and abs(c) >= threshold
            is_rep = (m == rep) and not incumbent_hit
            if incumbent_hit:
                redundant = f"incumbent:{name}"
            elif not is_rep:
                redundant = rep
            else:
                redundant = ""
            out[m] = {"cluster_id": rep, "cluster_rep": bool(is_rep), "redundant_with": redundant,
                      "max_corr": round(c, 3), "corr_with": f"{kind}:{name}" if name else "",
                      "cluster_size": len(members)}

    # residual t against the single most-correlated reference
    for cid in ids:
        name, c, kind = best[cid]
        arr = None
        if abs(c) >= RESIDUAL_MIN_CORR:
            arr = refs.get(name) if kind == "incumbent" else priors.get(name) if kind == "prior" else x.get(name)
        if arr is None:
            out[cid].update({"residual_dev_ic": np.nan, "residual_dev_t": t[cid], "residual_vs": ""})
            continue
        from evaluation.metrics import summarize_ic
        s = summarize_ic(residual_rank_ic(dates, x[cid], arr, label), nw_lags=NW_LAGS)
        if not s["n_days"] or not np.isfinite(s["t_stat"]):
            # rank-identical to its reference (e.g. rank(f) vs f): nothing is left
            s = {"ic_mean": 0.0, "t_stat": 0.0}
        out[cid].update({"residual_dev_ic": float(s["ic_mean"]), "residual_dev_t": float(s["t_stat"]),
                         "residual_vs": f"{kind}:{name}"})
    return out
