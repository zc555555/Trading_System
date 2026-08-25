"""Options implied-volatility PROTOTYPE on the free Massive (ex-Polygon) tier.

Goal of the prototype: prove the pipeline end-to-end on a handful of
names before paying for a plan -- contract discovery as-of a date,
daily option bars, Black-Scholes inversion to implied vol, and a daily
per-underlying IV summary (ATM IV, 30-day term, put-call IV skew).

Free-tier facts that shape the design (massive.com/pricing, 2026-08):
  * Basic: contracts reference + option OHLC bars, 2 years of history,
    5 API calls / minute, end-of-day. NO snapshot/greeks endpoint, and the
    snapshot is current-only on every plan -- historical IV must be
    computed from prices on any plan, so this code is not throwaway.
  * Budget: 5 calls/min => ~300/hour. The prototype pulls, per underlying
    and per month-end as-of date, the nearest-ATM call and put expiring
    ~30 days out (2 contracts), then their daily bars in one call each.

Outputs data/massive_iv_prototype.parquet with one row per (date,
symbol): iv_call, iv_put, iv_atm (mean), iv_skew (put - call), spot,
dte. Nothing here enters the feature pipeline until a pre-registered
factor experiment says so.

Usage:
    python data/fetch_massive_options.py --dry-run        # plan the calls, no network
    python data/fetch_massive_options.py --symbols AAPL,MSFT,JPM,XOM,PG --months 6
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.vendor_keys import get_key  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
PANEL = DATA_DIR / "stocks.parquet"
OUT = DATA_DIR / "massive_iv_prototype.parquet"
BASE = "https://api.massive.com"
CALLS_PER_MIN = 5
RISK_FREE = 0.04           # flat rf for the prototype; ^TNX-driven rf in the real version
TARGET_DTE = 30


class Throttle:
    def __init__(self, per_min: int):
        self.gap = 60.0 / per_min + 0.2
        self.last = 0.0
        self.calls = 0

    def wait(self):
        now = time.time()
        if now - self.last < self.gap:
            time.sleep(self.gap - (now - self.last))
        self.last = time.time()
        self.calls += 1


def bs_price(S, K, T, r, sigma, call: bool) -> float:
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, call: bool) -> float:
    """Bisection on sigma in [1%, 500%]; NaN if price is outside no-arbitrage bounds."""
    intrinsic = max(0.0, (S - K * math.exp(-r * T)) if call else (K * math.exp(-r * T) - S))
    if not (price > intrinsic + 1e-6) or T <= 0:
        return float("nan")
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def occ_ticker(symbol: str, expiry: date, call: bool, strike: float) -> str:
    return f"O:{symbol}{expiry.strftime('%y%m%d')}{'C' if call else 'P'}{int(round(strike * 1000)):08d}"


def list_contracts(sess, th, key, symbol: str, as_of: date, spot: float) -> pd.DataFrame:
    """Contracts live at as_of expiring 20-45 days later, near the money."""
    th.wait()
    r = sess.get(f"{BASE}/v3/reference/options/contracts", params={
        "underlying_ticker": symbol, "as_of": as_of.isoformat(),
        "expiration_date.gte": (as_of + timedelta(days=20)).isoformat(),
        "expiration_date.lte": (as_of + timedelta(days=45)).isoformat(),
        "strike_price.gte": round(spot * 0.85, 2), "strike_price.lte": round(spot * 1.15, 2),
        "limit": 1000, "apiKey": key}, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("results", []))


def daily_bars(sess, th, key, opt_ticker: str, start: date, end: date) -> pd.DataFrame:
    th.wait()
    r = sess.get(f"{BASE}/v2/aggs/ticker/{opt_ticker}/range/1/day/{start}/{end}",
                 params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": key},
                 timeout=60)
    r.raise_for_status()
    res = r.json().get("results", []) or []
    df = pd.DataFrame(res)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    return df[["date", "c", "v"]].rename(columns={"c": "close", "v": "volume"})


def main(symbols: list[str], months: int, dry_run: bool) -> int:
    key = get_key("MASSIVE_API_KEY")
    if not key and not dry_run:
        print("MASSIVE_API_KEY not set (env or config_keys.py). Free tier: massive.com -> sign up.")
        return 1
    prices = pd.read_parquet(PANEL, columns=["date", "symbol", "close"])
    prices["d"] = pd.to_datetime(prices["date"]).dt.date
    px = {s: g.set_index("d")["close"] for s, g in prices.groupby("symbol") if s in symbols}
    last = max(g.index.max() for g in px.values())
    as_ofs = [pd.Timestamp(last) - pd.DateOffset(months=k) for k in range(months, 0, -1)]
    as_ofs = [a.date() for a in as_ofs]
    plan = len(symbols) * len(as_ofs) * 3          # 1 contract list + 2 bar pulls
    print(f"symbols {symbols} | as-of dates {len(as_ofs)} ({as_ofs[0]} .. {as_ofs[-1]}) | "
          f"planned API calls {plan} (~{plan / CALLS_PER_MIN:.0f} min on the free tier)")
    if dry_run:
        return 0

    sess, th, rows = requests.Session(), Throttle(CALLS_PER_MIN), []
    for sym in symbols:
        series = px.get(sym)
        if series is None:
            print(f"  {sym}: not in price panel, skipped")
            continue
        for as_of in as_ofs:
            spot_idx = series.index[series.index <= as_of]
            if not len(spot_idx):
                continue
            spot = float(series.loc[spot_idx[-1]])
            try:
                cs = list_contracts(sess, th, key, sym, as_of, spot)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym} {as_of}: contracts failed ({exc})")
                continue
            if cs.empty:
                print(f"  {sym} {as_of}: no contracts")
                continue
            cs["moneyness"] = (cs["strike_price"] - spot).abs()
            picks = {}
            for ctype in ("call", "put"):
                sub = cs[cs["contract_type"] == ctype]
                if len(sub):
                    picks[ctype] = sub.sort_values(["moneyness", "expiration_date"]).iloc[0]
            for ctype, c in picks.items():
                expiry = pd.Timestamp(c["expiration_date"]).date()
                try:
                    bars = daily_bars(sess, th, key, c["ticker"], as_of, min(expiry, last))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {c['ticker']}: bars failed ({exc})")
                    continue
                for _, b in bars.iterrows():
                    d = b["date"]
                    if d not in series.index:
                        continue
                    T = (expiry - d).days / 365.0
                    iv = implied_vol(float(b["close"]), float(series.loc[d]), float(c["strike_price"]),
                                     T, RISK_FREE, ctype == "call")
                    rows.append({"date": d, "symbol": sym, "type": ctype, "strike": float(c["strike_price"]),
                                 "expiry": expiry, "dte": (expiry - d).days, "opt_close": float(b["close"]),
                                 "spot": float(series.loc[d]), "iv": iv, "volume": float(b["volume"])})
            print(f"  {sym} {as_of}: spot {spot:.2f}, {len(picks)} contracts, calls used {th.calls}")

    if not rows:
        print("no rows produced")
        return 1
    raw = pd.DataFrame(rows)
    wide = raw.pivot_table(index=["date", "symbol"], columns="type", values="iv", aggfunc="mean").reset_index()
    wide = wide.rename(columns={"call": "iv_call", "put": "iv_put"})
    wide["iv_atm"] = wide[["iv_call", "iv_put"]].mean(axis=1)
    wide["iv_skew"] = wide.get("iv_put", np.nan) - wide.get("iv_call", np.nan)
    meta = raw.groupby(["date", "symbol"])[["spot", "dte"]].first().reset_index()
    out = wide.merge(meta, on=["date", "symbol"])
    out.to_parquet(OUT, index=False)
    print(f"\nsaved {len(out)} (date, symbol) rows -> {OUT}")
    print(out.groupby("symbol")[["iv_atm", "iv_skew"]].describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT,JPM,XOM,PG")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.symbols.split(","), a.months, a.dry_run))
