"""Production feature chain on the mid-cap research panel (needed by the
full stage / pool releases, which train the incumbent factor groups).

Reuses evaluation/experiments/survivorship_universe.run_feature_chain_on:
the raw panel is swapped into data/stocks.parquet, prepare_prediction_data.py
runs unchanged, the featured panel is copied out, and every production file
is restored from backup unconditionally. NEVER run while the nightly task
(21:00 local) is active -- the chain and the task read the same files.

The raw panel is trimmed to each member's membership windows plus a
320-session warm-up (trailing features need history; 250 is the DSL's
maximum lookback): 5.6M -> ~3.4M rows, which keeps the chain and the
walk-forward within memory. Rows outside membership windows were never
point-in-time anyway.

Output: data/stocks_with_time_windows_midcap.parquet (OHLCV + production
features + label), replacing the OHLCV-only panel; the screen cache must
be rebuilt afterwards (harness.load_screen_panel(rebuild=True)).

Usage:  python data/build_midcap_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent
RESEARCH = DATA.parent
sys.path.append(str(RESEARCH))
PANEL = DATA / "stocks_with_time_windows_midcap.parquet"
OHLCV = DATA / "stocks_ohlcv_midcap.parquet"          # kept as the untouched raw panel
MEMBERSHIP = DATA / "midcap_membership.parquet"
WARMUP = 320


def trimmed_raw() -> pd.DataFrame:
    src = OHLCV if OHLCV.exists() else PANEL
    px = pd.read_parquet(src, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    if not OHLCV.exists():
        px.to_parquet(OHLCV, index=False)
    mem = pd.read_parquet(MEMBERSHIP)
    d = pd.to_datetime(px["date"]).dt.tz_localize(None)
    sessions = np.sort(d.unique())
    first = mem.groupby("symbol")["start"].min()
    last = mem.groupby("symbol")["end"].max()
    lo = first.reindex(px["symbol"]).to_numpy().astype("datetime64[ns]")
    hi = last.reindex(px["symbol"]).to_numpy()
    hi = np.where(pd.isna(hi), np.datetime64("2100-01-01"), hi.astype("datetime64[ns]"))
    pos = np.searchsorted(sessions, lo)
    warm = sessions[np.clip(pos - WARMUP, 0, len(sessions) - 1)]
    keep = (d.to_numpy() >= warm) & (d.to_numpy() <= hi)
    out = px[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"raw panel {len(px):,} rows -> {len(out):,} rows within membership + {WARMUP}-session warm-up "
          f"({out.symbol.nunique():,} symbols)", flush=True)
    return out


def main():
    from evaluation.experiments.survivorship_universe import run_feature_chain_on
    raw = trimmed_raw()
    out = run_feature_chain_on(raw, out_panel=PANEL)
    print(f"featured mid-cap panel: {len(out):,} rows, {len(out.columns)} columns -> {PANEL.name}")


if __name__ == "__main__":
    sys.exit(main())
