"""
Liquidity filter for the OHLCV parquet.

Runs AFTER fetch_ohlcv.py. Drops symbols that fail any of:
  - 20-day average dollar volume below ``min_dollar_volume_20d``
  - Latest close price below ``min_price``
  - History length below ``min_history_days``

Each symbol is evaluated on its most recent ``min_history_days`` trading
days, not the full series — that way a name that became illiquid only
during the early backfill window (e.g. a recent IPO) still survives.

Writes:
  - data/stocks.parquet                : filtered in-place
  - data/stocks.parquet.prefilter.bak  : one-time pre-filter backup (kept for diff)
  - data/liquidity_report.json         : per-symbol pass/fail + reason

CLI:
  python research/data/apply_liquidity_filter.py
  python research/data/apply_liquidity_filter.py --dry-run     # report only
  python research/data/apply_liquidity_filter.py --no-backup   # skip .bak file

Filter parameters come from research/config.yaml::data.liquidity_filter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Allow running as a script
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config_loader import load_config  # noqa: E402


def evaluate_symbol(
    sym_df: pd.DataFrame,
    min_dollar_volume: float,
    min_price: float,
    min_history_days: int,
) -> Tuple[bool, Dict]:
    """Return (passes, diagnostics_dict) for a single symbol."""
    n = len(sym_df)
    diag: Dict = {"rows": n}

    if n < min_history_days:
        diag.update(reason="insufficient_history", history_days=n)
        return False, diag

    recent = sym_df.tail(min_history_days)
    last_close = float(recent["close"].iloc[-1])
    diag["last_close"] = last_close

    if last_close < min_price:
        diag.update(reason="below_min_price")
        return False, diag

    # 20-day average $ volume on the most recent 20 days
    tail20 = recent.tail(20)
    dollar_vol = (tail20["close"] * tail20["volume"]).mean()
    diag["adv_20d_usd"] = float(dollar_vol)

    if dollar_vol < min_dollar_volume:
        diag.update(reason="below_min_dollar_volume")
        return False, diag

    diag["reason"] = "ok"
    return True, diag


def apply_liquidity_filter(
    parquet_path: Path,
    min_dollar_volume: float,
    min_price: float,
    min_history_days: int,
    dry_run: bool = False,
    backup: bool = True,
) -> Dict:
    """Apply the filter and (unless dry_run) overwrite the parquet."""
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    keep_symbols: List[str] = []
    drop_symbols: List[str] = []
    report: Dict[str, Dict] = {}

    grouped = df.groupby("symbol", sort=False)
    for symbol, sym_df in grouped:
        passes, diag = evaluate_symbol(
            sym_df,
            min_dollar_volume=min_dollar_volume,
            min_price=min_price,
            min_history_days=min_history_days,
        )
        report[symbol] = diag
        if passes:
            keep_symbols.append(symbol)
        else:
            drop_symbols.append(symbol)

    print(f"[liquidity] Evaluated {len(report)} symbols")
    print(f"[liquidity] Pass: {len(keep_symbols)}   Fail: {len(drop_symbols)}")
    if drop_symbols:
        # Group drops by reason for a quick summary
        from collections import Counter
        reasons = Counter(report[s]["reason"] for s in drop_symbols)
        for r, c in reasons.most_common():
            print(f"  [drop:{r}] {c} symbols")

    if dry_run:
        print("[liquidity] DRY RUN — parquet not modified")
    else:
        if backup:
            bak = parquet_path.with_suffix(parquet_path.suffix + ".prefilter.bak")
            if not bak.exists():
                shutil.copy2(parquet_path, bak)
                print(f"[liquidity] Backed up original to {bak.name}")
        filtered = df[df["symbol"].isin(keep_symbols)].reset_index(drop=True)
        filtered.to_parquet(parquet_path, index=False, compression="snappy")
        print(f"[liquidity] Wrote filtered parquet: {len(filtered):,} rows, "
              f"{len(keep_symbols)} symbols")

    return {
        "kept": keep_symbols,
        "dropped": drop_symbols,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Apply liquidity filter to stocks.parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report only; do not modify the parquet.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing the .prefilter.bak file.")
    args = parser.parse_args()

    config = load_config()
    lf = config["data"].get("liquidity_filter", {})

    if not lf.get("enabled", True):
        print("[liquidity] Filter disabled in config.yaml (data.liquidity_filter.enabled)")
        return

    parquet_path = Path(__file__).parent / config["data"].get("parquet_path", "stocks.parquet")
    if not parquet_path.exists():
        print(f"[liquidity] ERROR: parquet not found at {parquet_path}")
        sys.exit(1)

    print("=" * 70)
    print("LIQUIDITY FILTER")
    print("=" * 70)
    print(f"Source           : {parquet_path}")
    print(f"min_dollar_volume_20d: ${lf.get('min_dollar_volume_20d', 0):,.0f}")
    print(f"min_price        : ${lf.get('min_price', 0):.2f}")
    print(f"min_history_days : {lf.get('min_history_days', 0)}")
    print()

    result = apply_liquidity_filter(
        parquet_path=parquet_path,
        min_dollar_volume=lf.get("min_dollar_volume_20d", 50_000_000),
        min_price=lf.get("min_price", 5.0),
        min_history_days=lf.get("min_history_days", 252),
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    report_path = parquet_path.parent / "liquidity_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "kept_count": len(result["kept"]),
                "dropped_count": len(result["dropped"]),
                "kept": sorted(result["kept"]),
                "dropped": sorted(result["dropped"]),
                "report": result["report"],
            },
            f,
            indent=2,
        )
    print(f"[liquidity] Per-symbol report -> {report_path}")


if __name__ == "__main__":
    main()
