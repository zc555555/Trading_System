"""Backfill historical earnings dates + surprises via yfinance (Layer 4).

Per symbol: announcement timestamps and EPS surprise(%) back to ~2002
(about 100 quarterly rows). Future scheduled announcements are kept --
downstream feature code must treat them as unavailable until they occur.

Output: data/earnings_dates.parquet [symbol, earnings_ts, eps_estimate,
eps_reported, surprise_pct]. Resumable: symbols already present are
skipped; re-run to refresh (--refresh-days N re-fetches symbols whose
newest row is older than N days).

Usage:
    python data/fetch_earnings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config  # noqa: E402

OUT = Path(__file__).parent / "earnings_dates.parquet"
SLEEP_S = 0.4


def fetch_symbol(sym: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        ed = yf.Ticker(sym).get_earnings_dates(limit=100)  # Yahoo hard cap
    except Exception as e:
        print(f"  [warn] {sym}: {e}")
        return None
    if ed is None or ed.empty:
        return None
    df = ed.reset_index()
    df.columns = [c.strip() for c in df.columns]
    rename = {"Earnings Date": "earnings_ts", "EPS Estimate": "eps_estimate",
              "Reported EPS": "eps_reported", "Surprise(%)": "surprise_pct"}
    df = df.rename(columns=rename)
    keep = [c for c in ("earnings_ts", "eps_estimate", "eps_reported",
                        "surprise_pct") if c in df.columns]
    df = df[keep].copy()
    df["symbol"] = sym
    return df


def main():
    config = load_config()
    symbols = config["data"]["symbols"]

    frames, done = [], set()
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        frames.append(existing)
        done = set(existing["symbol"].unique())
        print(f"[resume] {len(done)} symbols cached")

    todo = [s for s in symbols if s not in done]
    print(f"[fetch] {len(todo)} symbols to go")
    for i, sym in enumerate(todo, 1):
        df = fetch_symbol(sym)
        if df is not None:
            frames.append(df)
            status = f"{len(df)} announcements"
        else:
            status = "NO DATA"
        if i % 25 == 0 or i == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
            print(f"  [{i}/{len(todo)}] {sym}: {status} (checkpoint saved)")
        time.sleep(SLEEP_S)

    final = pd.concat(frames, ignore_index=True)
    final.to_parquet(OUT, index=False)
    n_surp = final["surprise_pct"].notna().sum()
    print(f"\n[done] {final['symbol'].nunique()} symbols, {len(final):,} "
          f"announcements ({n_surp:,} with surprise) -> {OUT.name}")


if __name__ == "__main__":
    main()
