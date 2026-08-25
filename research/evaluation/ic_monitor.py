"""Factor IC decay monitor (pro-upgrade item 4) -- the third detection layer.

Layer 1 (weekly retrain) re-weights factors by recent rank IC and zeroes
negatives; layer 2 (pre-registered removal triggers) checks fixed dates.
Neither watches the months in between. This monitor does: it tracks each
production factor's realized 20-day rank IC as a rolling series, compares
it with the factor's own development-period distribution, and raises a
flag when a factor drifts out of its normal range for long enough that
noise is an unlikely explanation.

Data sources (stitched, earliest first):
  * history : the 7-factor walk-forward panel (oos_predictions_h20_cand_
              high52.parquet) -- per-date factor scores + matured labels
  * live    : nightly full-universe factor-score dumps written by
              get_daily_signals_multi_factor.py (artifacts/factor_scores/),
              labelled with log(close[t+20] / close[t]) from the price
              panel once 20 sessions have passed (same label definition
              as the walk-forward).

PRE-REGISTERED ALERT RULES (declared before any live reading was seen):
  WARN  : rolling-120-session IC below (dev mean - 1 x dev std of the
          rolling-120 series) for >= 20 consecutive sessions.
  ALERT : rolling-120-session IC below zero for >= 60 consecutive sessions,
          for a factor whose current production weight is > 0.
  An ALERT is a removal-review trigger, not an automatic removal: the
  decision still goes through the ledger.

Descriptive only: no orders, no weight changes. Runs weekly after retrain.

Usage:
    python evaluation/ic_monitor.py            # report + chart + json
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

RESEARCH = Path(__file__).resolve().parent.parent
RESULTS = RESEARCH / "evaluation" / "results"
HISTORY_PANEL = RESULTS / "oos_predictions_h20_cand_high52.parquet"
SCORES_DIR = RESEARCH / "artifacts" / "factor_scores"
WEIGHTS_PATH = RESEARCH / "artifacts" / "factor_weights.json"
PRICES = RESEARCH / "data" / "stocks.parquet"
IMG = RESEARCH.parent / "docs" / "img" / "ic_monitor.png"
WARNINGS_LOG = RESEARCH.parent / "trading_logs" / "WARNINGS.log"
NOTIFY_PS1 = RESEARCH.parent / "scripts" / "notify_failure.ps1"

HORIZON = 20
ROLL = 120
MIN_NAMES = 20
DEV_START, DEV_END = "2022-01-01", "2025-07-01"
WARN_SESSIONS = 20
ALERT_SESSIONS = 60


# --------------------------------------------------------------------------
# pure functions (unit-tested)
# --------------------------------------------------------------------------
def daily_factor_ic(panel: pd.DataFrame, factor_cols: list[str],
                    label_col: str, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Per-date Spearman rank IC of each factor column vs the label."""
    df = panel[["date"] + factor_cols + [label_col]].dropna(subset=[label_col])
    out = {}
    for c in factor_cols:
        sub = df[["date", c, label_col]].dropna()
        g = sub.groupby("date")
        rx = g[c].rank()
        ry = g[label_col].rank()
        tmp = pd.DataFrame({"date": sub["date"].values, "rx": rx.values, "ry": ry.values})
        gg = tmp.groupby("date")
        n = gg["rx"].count()
        # Pearson correlation of ranks == Spearman
        cov = gg.apply(lambda x: np.cov(x["rx"], x["ry"], ddof=0)[0, 1]
                       if len(x) >= 2 else np.nan, include_groups=False)
        sx = gg["rx"].std(ddof=0)
        sy = gg["ry"].std(ddof=0)
        ic = cov / (sx * sy)
        ic[n < min_names] = np.nan
        out[c] = ic
    return pd.DataFrame(out).sort_index()


def dev_baseline(rolling: pd.DataFrame, dev_start=DEV_START, dev_end=DEV_END) -> pd.DataFrame:
    idx = pd.DatetimeIndex(rolling.index)
    s, e = pd.Timestamp(dev_start), pd.Timestamp(dev_end)
    if idx.tz is not None:
        s, e = s.tz_localize(idx.tz), e.tz_localize(idx.tz)
    dev = rolling[(idx >= s) & (idx < e)]
    return pd.DataFrame({"mean": dev.mean(), "std": dev.std()})


def _trailing_run(mask: pd.Series) -> int:
    """Length of the run of True values ending at the last observation."""
    vals = mask.fillna(False).to_numpy(dtype=bool)
    n = 0
    for v in vals[::-1]:
        if not v:
            break
        n += 1
    return n


def evaluate_alerts(rolling: pd.DataFrame, baseline: pd.DataFrame,
                    weights: dict, warn_sessions: int = WARN_SESSIONS,
                    alert_sessions: int = ALERT_SESSIONS) -> dict:
    """Apply the pre-registered rules to the latest rolling-IC readings."""
    report = {}
    for f in rolling.columns:
        series = rolling[f].dropna()
        if (series.empty or f not in baseline.index
                or not np.isfinite(baseline.loc[f, "std"])):
            report[f] = {"status": "NO_DATA"}
            continue
        mean, std = float(baseline.loc[f, "mean"]), float(baseline.loc[f, "std"])
        run_band = _trailing_run(series < (mean - std))
        run_zero = _trailing_run(series < 0)
        w = float(weights.get(f.replace("factor_", ""), 0.0))
        status = "OK"
        if run_band >= warn_sessions:
            status = "WARN"
        if run_zero >= alert_sessions and w > 0:
            status = "ALERT"
        report[f] = {
            "status": status,
            "latest_rolling_ic": round(float(series.iloc[-1]), 5),
            "dev_mean": round(mean, 5), "dev_std": round(std, 5),
            "sessions_below_band": int(run_band),
            "sessions_below_zero": int(run_zero),
            "production_weight": round(w, 4),
            "as_of": str(pd.Timestamp(series.index[-1]).date()),
        }
    return report


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def load_history() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(HISTORY_PANEL)
    factor_cols = [c for c in panel.columns if c.startswith("factor_")
                   and panel[c].notna().any() and panel[c].nunique() > 1]
    return panel, factor_cols


def load_live(factor_cols: list[str], after) -> pd.DataFrame | None:
    """Stitch nightly score dumps with matured labels from the price panel."""
    files = sorted(SCORES_DIR.glob("factor_scores_*.parquet")) if SCORES_DIR.exists() else []
    if not files:
        return None
    scores = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    scores["date"] = pd.to_datetime(scores["date"])
    prices = pd.read_parquet(PRICES, columns=["date", "symbol", "close"])
    prices = prices.sort_values(["symbol", "date"])
    ptz = prices["date"].dt.tz
    if ptz is not None and scores["date"].dt.tz is None:
        scores["date"] = scores["date"].dt.tz_localize(ptz)
    elif ptz is not None:
        scores["date"] = scores["date"].dt.tz_convert(ptz)
    prices["future_return_20d"] = prices.groupby("symbol")["close"].transform(
        lambda x: np.log(x.shift(-HORIZON) / x))
    live = scores.merge(prices[["date", "symbol", "future_return_20d"]],
                        on=["date", "symbol"], how="left")
    live = live[live["date"] > after].dropna(subset=["future_return_20d"])
    keep = [c for c in factor_cols if c in live.columns]
    if not len(live) or not keep:
        return None
    return live[["date", "symbol"] + keep + ["future_return_20d"]]


def make_chart(rolling: pd.DataFrame, baseline: pd.DataFrame, report: dict,
               history_end, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = list(rolling.columns)
    n = len(cols)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 2.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, cols):
        s = rolling[f]
        ax.plot(s.index, s, lw=1.1, color="#4878CF")
        if f in baseline.index and np.isfinite(baseline.loc[f, "std"]):
            m, sd = baseline.loc[f, "mean"], baseline.loc[f, "std"]
            ax.axhspan(m - sd, m + sd, color="#4878CF", alpha=0.10)
            ax.axhline(m, color="#4878CF", lw=0.7, ls="--")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(history_end, color="gray", lw=0.8, ls=":")
        st = report.get(f, {}).get("status", "")
        color = {"OK": "green", "WARN": "orange", "ALERT": "red"}.get(st, "gray")
        ax.set_title(f"{f.replace('factor_', '')}  [{st}]", fontsize=10, color=color)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Rolling {ROLL}-session rank IC per factor "
                 f"(band = dev mean +/- 1 sd; dotted = history ends, live dumps begin)",
                 fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def notify(title: str, body: str) -> None:
    if sys.platform.startswith("win") and NOTIFY_PS1.exists():
        try:
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", str(NOTIFY_PS1), title, body],
                           timeout=30, check=False)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    print(f"=== IC decay monitor ({datetime.now():%Y-%m-%d %H:%M}) ===")
    panel, factor_cols = load_history()
    print(f"history panel: {len(panel):,} rows, factors: "
          f"{[c.replace('factor_', '') for c in factor_cols]}")
    ics = daily_factor_ic(panel, factor_cols, "future_return_20d")
    history_end = ics.index.max()

    live = load_live(factor_cols, after=history_end)
    n_live = 0
    if live is not None:
        live_ics = daily_factor_ic(live, [c for c in factor_cols if c in live.columns],
                                   "future_return_20d")
        n_live = int(len(live_ics))
        ics = pd.concat([ics, live_ics]).sort_index()
        print(f"live dumps: {n_live} matured sessions after {history_end.date()}")
    else:
        print(f"live dumps: none matured yet (history ends {history_end.date()}; "
              f"nightly dumps accumulate from today, mature after {HORIZON} sessions)")

    rolling = ics.rolling(ROLL, min_periods=int(ROLL * 0.67)).mean()
    baseline = dev_baseline(rolling)
    # artifacts/factor_weights.json: the production blend uses
    # "effective_weights" (IC-derived, negatives zeroed at the last retrain).
    weights = {}
    if WEIGHTS_PATH.exists():
        raw = json.load(open(WEIGHTS_PATH, encoding="utf-8"))
        weights = raw.get("effective_weights") or raw.get("weights") or {
            k: v for k, v in raw.items() if isinstance(v, (int, float))}
    report = evaluate_alerts(rolling, baseline, weights)

    print(f"\n{'factor':<12}{'status':<8}{'roll120':>9}{'dev_mean':>9}{'dev_sd':>8}"
          f"{'<band':>7}{'<0':>5}{'weight':>8}")
    print("-" * 66)
    for f, r in report.items():
        name = f.replace("factor_", "")
        if r["status"] == "NO_DATA":
            print(f"{name:<12}{'NO_DATA':<8}")
            continue
        print(f"{name:<12}{r['status']:<8}{r['latest_rolling_ic']:>9.4f}"
              f"{r['dev_mean']:>9.4f}{r['dev_std']:>8.4f}{r['sessions_below_band']:>7}"
              f"{r['sessions_below_zero']:>5}{r['production_weight']:>8.3f}")

    out = {"generated_at": datetime.now().isoformat(), "roll": ROLL, "horizon": HORIZON,
           "rules": {"warn": f"rolling IC < dev_mean - 1sd for >= {WARN_SESSIONS} sessions",
                     "alert": f"rolling IC < 0 for >= {ALERT_SESSIONS} sessions and weight > 0"},
           "history_end": str(history_end.date()), "live_sessions": n_live,
           "factors": report}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "ic_monitor_latest.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    rolling.to_csv(RESULTS / "ic_monitor_rolling.csv")
    make_chart(rolling, baseline, report, history_end, IMG)
    print(f"\nsaved: ic_monitor_latest.json, ic_monitor_rolling.csv, {IMG.name}")

    flagged = {f: r for f, r in report.items() if r.get("status") in ("WARN", "ALERT")}
    if flagged:
        WARNINGS_LOG.parent.mkdir(exist_ok=True)
        line = "; ".join(f"{f.replace('factor_', '')}={r['status']}"
                         f"(roll={r['latest_rolling_ic']:+.4f})" for f, r in flagged.items())
        with open(WARNINGS_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M} IC MONITOR {line}\n")
        print(f"\n[FLAG] {line} -> {WARNINGS_LOG}")
        if any(r["status"] == "ALERT" for r in flagged.values()):
            notify("StockPredict factor IC decay ALERT", line)
    else:
        print("\nno factor outside its pre-registered band")
    return 0


if __name__ == "__main__":
    # UTF-8 stdout only when run as a script (cp1252 console pipes); doing
    # this at import time would break pytest's capture.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
