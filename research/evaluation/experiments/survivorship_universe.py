"""Survivorship-complete universe: re-measure the baseline with the dead included.

The free price panel (yfinance) only contains companies that still
exist, so every S&P member that was later delisted -- bankruptcies,
take-unders, the names that make survivorship bias bite -- is missing
from every evaluation so far. Sharadar SEP now provides their histories
(data/fetch_sharadar_prices.py; delisted tickers resolved through
TICKERS.relatedtickers). This experiment adds ONLY the departed members
(membership table rows with an end date) to the panel, so the contrast
isolates survivorship; universe breadth (the ~230 current members the
yfinance panel never had) is a separate question for a separate test.

Design (a MEASUREMENT correction, not a hypothesis; nothing is adopted
or rejected -- the result is a corrected ruler):
  1. extended raw panel = stocks.parquet UNION Sharadar rows for departed
     members not already present; Sharadar O/H/L/C are scaled by
     closeadj/close so both sources share yfinance's dividend+split-
     adjusted convention;
  2. run the PRODUCTION feature chain on it (prepare_prediction_data.py,
     unmodified) by swapping stocks.parquet in and restoring every
     production file afterwards -- always, even on failure;
  3. two identical h=20 walk-forwards under the point-in-time membership
     mask: 'pitCUR' (current panel, dead missing) and 'surv' (extended);
  4. report the four-tier blended rank IC side by side.
PRE-REGISTERED EXPECTATION (written before running): survivorship bias
inflates measured IC, so 'surv' should come in at or below 'pitCUR' in
the virgin and dev tiers; if coverage of departed members is >= 80%, the
'surv' panel becomes the evaluation standard going forward regardless
of which direction the numbers move.

Usage:
    python evaluation/experiments/survivorship_universe.py --max-folds 1   # smoke
    python evaluation/experiments/survivorship_universe.py
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.purged_walk_forward import (  # noqa: E402
    WalkForwardConfig, run_walk_forward, DATA_PATH, RESULTS_DIR)
from evaluation.experiments.pit_universe_test import (  # noqa: E402
    membership_mask, compare, MEMBERSHIP)

RESEARCH = Path(__file__).resolve().parent.parent.parent
DATA = RESEARCH / "data"
SHARADAR = DATA / "sharadar_prices.parquet"
RAW = DATA / "stocks.parquet"
PROD_FILES = ["stocks.parquet", "stocks_features.parquet",
              "stocks_selected_features.parquet", "stocks_with_time_windows.parquet"]
BACKUP = DATA / "_backup_survivorship"
SURV_PANEL = DATA / "stocks_with_time_windows_surv.parquet"
START = "2014-01-01"
HORIZON = 20


def departed_members() -> pd.DataFrame:
    iv = pd.read_parquet(MEMBERSHIP)
    # only departures inside the evaluation window matter; earlier ones
    # have no rows after START and risk ticker-reuse contamination
    return iv[iv["end"].notna() & (iv["end"] >= pd.Timestamp(START))]


def build_extended_raw() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(RAW)
    have = set(raw["symbol"].unique())
    dep = departed_members()
    dep_syms = sorted(set(dep["symbol"]) - have)
    sh = pd.read_parquet(SHARADAR)
    sh = sh[sh["symbol"].isin(dep_syms)].copy()
    covered = sorted(sh["symbol"].unique())
    missing = sorted(set(dep_syms) - set(covered))

    sh["date"] = pd.to_datetime(sh["date"])
    tz = raw["date"].dt.tz
    if tz is not None:
        sh["date"] = sh["date"].dt.tz_localize(tz)
    sh = sh[sh["date"] >= pd.Timestamp(START, tz=tz)]
    # align adjustment convention with yfinance (dividend + split adjusted)
    factor = (sh["closeadj"] / sh["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    add = pd.DataFrame({
        "date": sh["date"], "symbol": sh["symbol"],
        "open": sh["open"] * factor, "high": sh["high"] * factor,
        "low": sh["low"] * factor, "close": sh["closeadj"],
        "volume": sh["volume"],
    })
    add = add.dropna(subset=["close"]).sort_values(["symbol", "date"])
    ext = pd.concat([raw[["date", "symbol", "open", "high", "low", "close", "volume"]], add],
                    ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    info = {"departed_members": len(dep_syms), "covered_by_sharadar": len(covered),
            "coverage": round(len(covered) / max(len(dep_syms), 1), 3),
            "missing_examples": missing[:20], "added_rows": int(len(add)),
            "panel_symbols_before": len(have), "panel_symbols_after": int(ext["symbol"].nunique())}
    return ext, info


def run_feature_chain_on(ext_raw: pd.DataFrame) -> pd.DataFrame:
    """Swap the extended raw panel in, run the production chain, take the
    result, restore production files unconditionally."""
    BACKUP.mkdir(exist_ok=True)
    for f in PROD_FILES:
        src = DATA / f
        if src.exists():
            shutil.copy2(src, BACKUP / f)
    try:
        ext_raw.to_parquet(RAW, index=False)
        print("running production feature chain on the extended panel (10-20 min)...")
        subprocess.run([sys.executable, "prepare_prediction_data.py"], cwd=RESEARCH, check=True)
        out = pd.read_parquet(DATA_PATH)
        out.to_parquet(SURV_PANEL, index=False)
        return out
    finally:
        for f in PROD_FILES:
            b = BACKUP / f
            if b.exists():
                shutil.copy2(b, DATA / f)
        print("production data files restored from backup")


def main(max_folds: int | None, reuse_panel: bool) -> None:
    print("=== survivorship-complete universe experiment ===")
    if reuse_panel and SURV_PANEL.exists():
        surv = pd.read_parquet(SURV_PANEL)
        info = {"reused": str(SURV_PANEL)}
    else:
        ext_raw, info = build_extended_raw()
        print(json.dumps(info, indent=1))
        surv = run_feature_chain_on(ext_raw)
    cur = pd.read_parquet(DATA_PATH)

    iv = pd.read_parquet(MEMBERSHIP)
    members = set(iv["symbol"].unique())
    runs = {}
    for tag, df in (("pitCUR", cur), ("surv", surv)):
        sub = df[df["symbol"].isin(members)].copy()
        mask = membership_mask(sub)
        data = sub[mask].copy()
        print(f"\n[{tag}] {sub['symbol'].nunique()} member symbols, PIT keeps {mask.mean():.1%} "
              f"of rows -> {data['symbol'].nunique()} symbols, {len(data):,} rows")
        runs[tag] = run_walk_forward(WalkForwardConfig(horizon=HORIZON, min_tail_test=15),
                                     max_folds=max_folds, df=data, tag=tag)
    if not max_folds:
        compare("pitCUR", "surv")
    report = {"generated_at": datetime.now().isoformat(), "panel_info": info,
              "horizon": HORIZON, "max_folds": max_folds}
    with open(RESULTS_DIR / "survivorship_universe_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nsaved: {RESULTS_DIR / 'survivorship_universe_report.json'}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--reuse-panel", action="store_true",
                    help="skip the feature chain and reuse stocks_with_time_windows_surv.parquet")
    a = ap.parse_args()
    main(a.max_folds, a.reuse_panel)
