"""
Stock universe definitions.

Single source of truth for which symbols enter the training/prediction pipeline.

Universes:
    sp100       : ~100 mega-cap names (hardcoded, always available offline)
    sp500       : Full S&P 500, fetched live from Wikipedia (cached locally)
    sp500_safe  : sp100 + ~100 high-conviction large/mid-caps (offline-safe)
    benchmarks  : Broad-market ETFs (SPY/QQQ/IWM/DIA + sector ETFs)
    all         : sp500 + benchmarks  (recommended for production)

Conventions
-----------
Tickers use the yfinance convention (e.g. BRK-B, BF-B). The downstream
trading layer (Alpaca) accepts these directly.

The Wikipedia fetcher writes a cached snapshot to:
    research/data/cache/sp500_constituents.csv
so an offline rebuild reuses the last known list.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
import sys


# ---------------------------------------------------------------------------
# Hardcoded S&P 100 — mega-cap, stable, yfinance-compatible tickers.
# This list is the offline-safe core. It should not need frequent updates.
# ---------------------------------------------------------------------------
SP100: List[str] = [
    # Mega-cap tech / communication
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "INTC", "AMD", "IBM",
    "NFLX", "TXN", "AMAT", "INTU", "NOW", "QCOM", "MU",
    "VZ", "T", "CMCSA", "DIS", "TMUS",

    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "DHR",
    "PFE", "BMY", "GILD", "SYK", "ISRG", "ELV", "REGN", "ZTS", "BDX",
    "BSX", "CI", "VRTX", "MDT",

    # Financials
    "JPM", "V", "MA", "BAC", "GS", "BLK", "MS", "AXP", "WFC", "C",
    "SCHW", "CB", "MMC", "USB", "COF", "PNC", "ICE", "SPGI", "CME",
    "BX", "AON",

    # Consumer Discretionary
    "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",

    # Consumer Staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "TGT",

    # Industrials
    "CAT", "GE", "RTX", "HON", "DE", "LMT", "BA", "UPS", "UNP", "ETN",
    "ADP", "MMM",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",

    # Utilities
    "NEE", "DUK", "SO",

    # Materials
    "LIN", "SHW", "APD",

    # Real Estate
    "PLD",
]
assert len(set(SP100)) == len(SP100), "SP100 contains duplicates"


# ---------------------------------------------------------------------------
# Additional reliable large/mid-caps used when Wikipedia is unreachable.
# These extend SP100 to ~200 high-quality, high-liquidity tickers.
# ---------------------------------------------------------------------------
SP500_SAFE_EXTRA: List[str] = [
    # Tech / Software
    "PANW", "SNPS", "CDNS", "FTNT", "ADSK", "WDAY", "ANET", "DELL",
    "HPQ", "HPE", "MSI", "ROP", "GEN", "ZBRA", "NET", "DDOG", "CRWD",
    "MDB", "SNOW", "TEAM", "OKTA", "ZS", "MRVL", "ON", "MPWR", "LRCX",
    "KLAC", "ASML", "ARM",

    # Healthcare extras
    "MCK", "CVS", "HCA", "HUM", "CNC", "DXCM", "EW", "IDXX", "WST",
    "RMD", "ALGN", "MTD", "IQV", "WAT", "ZBH", "BAX", "BIIB", "GEHC",

    # Financials extras
    "PYPL", "FI", "FIS", "AIG", "PRU", "MET", "ALL", "TRV", "AFL",
    "AMP", "DFS", "STT", "BK", "NTRS", "MTB", "FITB", "RF", "HBAN",
    "KEY", "WTW", "MCO", "MSCI", "NDAQ",

    # Consumer Discretionary extras
    "MAR", "HLT", "ABNB", "DRI", "CMG", "YUM", "ORLY", "AZO", "ROST",
    "DG", "DLTR", "BBY", "F", "GM", "LVS", "MGM",

    # Consumer Staples extras
    "KMB", "GIS", "K", "HSY", "STZ", "MNST", "KDP", "KHC", "CHD",
    "MKC", "CLX",

    # Industrials extras
    "EMR", "ITW", "PH", "ROK", "GD", "NOC", "TDG", "HEI", "FDX",
    "WM", "RSG", "CSX", "NSC", "ODFL", "JBHT", "PCAR", "URI", "AME",
    "OTIS", "DOV", "PWR", "TT", "CARR",

    # Energy extras
    "PSX", "VLO", "MPC", "OXY", "HES", "DVN", "FANG", "WMB", "KMI",
    "OKE", "TRGP", "BKR", "HAL",

    # Utilities extras
    "AEP", "SRE", "XEL", "D", "EXC", "ED", "WEC", "ES", "PCG", "PEG",
    "AWK", "PPL", "CMS", "DTE",

    # Materials extras
    "FCX", "NEM", "ECL", "DOW", "DD", "PPG", "NUE", "CTVA", "LYB",
    "STLD", "VMC", "MLM",

    # Real Estate extras
    "AMT", "EQIX", "PSA", "O", "WELL", "CCI", "DLR", "SPG", "VICI",
    "EXR", "AVB", "EQR", "INVH",

    # Communication extras
    "CHTR", "EA", "TTWO", "WBD", "PARA",
]
assert len(set(SP500_SAFE_EXTRA)) == len(SP500_SAFE_EXTRA), "SP500_SAFE_EXTRA contains duplicates"
assert not (set(SP100) & set(SP500_SAFE_EXTRA)), "SP500_SAFE_EXTRA overlaps SP100"


# ---------------------------------------------------------------------------
# Benchmark / sector ETFs (used as both training universe candidates AND
# as features-source for market context, see build_dataset.compute_market_features).
# ---------------------------------------------------------------------------
BENCHMARK_ETFS: List[str] = [
    # Broad market
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "DIA",   # Dow Jones
    "IWM",   # Russell 2000 (small cap)
    "VTI",   # Total US market
    # Sector ETFs (SPDR Select Sector)
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
    # Other
    "TLT",   # 20Y Treasury
    "GLD",   # Gold
    "USO",   # Oil
    "UUP",   # Dollar Index
]


# ---------------------------------------------------------------------------
# Wikipedia fetcher with local cache
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(__file__).parent / "cache"
_SP500_CACHE = _CACHE_DIR / "sp500_constituents.csv"
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _normalize_ticker(ticker: str) -> str:
    """Convert Wikipedia ticker format to yfinance format (BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def fetch_sp500_from_wikipedia(use_cache: bool = True) -> List[str]:
    """Fetch the current S&P 500 constituents from Wikipedia.

    On network failure, falls back to the cached CSV if available.
    On total failure, returns SP100 + SP500_SAFE_EXTRA.
    """
    import pandas as pd

    if use_cache and _SP500_CACHE.exists():
        try:
            df = pd.read_csv(_SP500_CACHE)
            tickers = [_normalize_ticker(t) for t in df["Symbol"].tolist()]
            print(f"[universe] Loaded {len(tickers)} S&P 500 tickers from cache: {_SP500_CACHE.name}")
            return tickers
        except Exception as e:
            print(f"[universe] Cache read failed ({e}), refetching...", file=sys.stderr)

    try:
        tables = pd.read_html(_WIKI_URL)
        df = tables[0]  # First table is the constituents
        if "Symbol" not in df.columns:
            raise ValueError(f"Wikipedia table schema changed; columns = {list(df.columns)}")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_SP500_CACHE, index=False)
        tickers = [_normalize_ticker(t) for t in df["Symbol"].tolist()]
        print(f"[universe] Fetched {len(tickers)} S&P 500 tickers from Wikipedia -> cached")
        return tickers
    except Exception as e:
        print(f"[universe] Wikipedia fetch failed ({e}), falling back to offline list", file=sys.stderr)
        return SP100 + SP500_SAFE_EXTRA


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def get_universe(name: str, include_benchmarks: bool = False) -> List[str]:
    """Resolve a universe name to a list of tickers.

    Args:
        name: One of {"sp100", "sp500", "sp500_safe", "benchmarks", "all"}.
        include_benchmarks: If True, append BENCHMARK_ETFS to the result.

    Returns:
        Deduplicated, sorted list of yfinance-compatible tickers.
    """
    name = name.lower().strip()

    if name == "sp100":
        tickers = list(SP100)
    elif name == "sp500_safe":
        tickers = SP100 + SP500_SAFE_EXTRA
    elif name == "sp500":
        tickers = fetch_sp500_from_wikipedia()
    elif name == "benchmarks":
        tickers = list(BENCHMARK_ETFS)
        include_benchmarks = False  # already included
    elif name == "all":
        tickers = fetch_sp500_from_wikipedia()
        include_benchmarks = True
    else:
        raise ValueError(
            f"Unknown universe '{name}'. "
            f"Valid: sp100, sp500, sp500_safe, benchmarks, all"
        )

    if include_benchmarks:
        tickers = tickers + BENCHMARK_ETFS

    # Dedupe, preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def universe_summary() -> None:
    """Print a summary of all available universes (for CLI use)."""
    for name in ("sp100", "sp500_safe", "sp500", "benchmarks", "all"):
        try:
            tickers = get_universe(name)
            print(f"  {name:<14} {len(tickers):>4} tickers")
        except Exception as e:
            print(f"  {name:<14}  ERROR  {e}")


if __name__ == "__main__":
    print("Available universes:")
    universe_summary()
    print(f"\nSP100 sample (first 20): {SP100[:20]}")
