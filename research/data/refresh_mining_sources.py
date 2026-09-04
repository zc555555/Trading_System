"""Refresh the mining DSL's external sources (idempotent; run monthly).

    1. SEC Insider Transactions data sets: download every quarterly zip not
       yet in data/form345/ (the SEC posts a quarter roughly one quarter
       late), then rebuild form4_daily.parquet when anything was new.
    2. FINRA consolidated short interest: full refetch (~25 min; the API has
       no incremental filter worth the complexity).
    3. GDELT: rescan the current year (and the previous one in January) with
       the full name table; earlier years are kept.
    4. Rebuild both mining screen caches so the next run sees the new data.

Nothing here touches production: the nightly pipeline does not read these
files (see PRODUCTION.md). Exit code 0 unless a step raised.

Usage:  python data/refresh_mining_sources.py [--skip sec,finra,gdelt,caches]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent
RESEARCH = DATA.parent
PY = sys.executable
UA = {"User-Agent": "Trading_System research zhaochenwu666@gmail.com"}
SEC_PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
FORM345 = DATA / "form345"
FIRST_QUARTER = "2014q1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refresh_sec() -> bool:
    """Download missing quarterly zips; return True when something was new."""
    FORM345.mkdir(exist_ok=True)
    html = requests.get(SEC_PAGE, headers=UA, timeout=60).text
    links = sorted(set(re.findall(r'href="([^"]*?/(\d{4}q[1-4])_form345\.zip)"', html)))
    new = False
    for href, quarter in links:
        if quarter < FIRST_QUARTER:
            continue
        target = FORM345 / f"{quarter}_form345.zip"
        if target.exists() and target.stat().st_size > 0:
            continue
        url = href if href.startswith("http") else "https://www.sec.gov" + href
        r = requests.get(url, headers=UA, timeout=300)
        if r.status_code != 200 or len(r.content) < 1000:
            log(f"  SEC {quarter}: HTTP {r.status_code}, skipped")
            continue
        target.write_bytes(r.content)
        log(f"  SEC {quarter}: downloaded {len(r.content) / 1e6:.1f} MB")
        new = True
        time.sleep(0.5)
    if new:
        subprocess.run([PY, str(DATA / "build_form4_fields.py")], check=True)
    else:
        log("  SEC: no new quarter")
    return new


def refresh_finra() -> None:
    subprocess.run([PY, str(DATA / "fetch_finra_short_interest.py"), "--sleep", "0.3"], check=True)


def refresh_gdelt() -> None:
    today = date.today()
    years = [today.year] if today.month > 1 else [today.year - 1, today.year]
    subprocess.run([PY, str(DATA / "gdelt_bigquery.py"),
                    "--names", str(DATA / "company_names_full.json"),
                    "--years", *[str(y) for y in years]], check=True)


def rebuild_caches() -> None:
    code = ("import sys; sys.path.insert(0, r'%s'); from mining import harness\n"
            "for h in (20, 5):\n"
            "    harness.configure(h); df = harness.load_screen_panel(rebuild=True)\n"
            "    print('cache h=%%d rows=%%d cols=%%d' %% (h, len(df), len(df.columns)))\n" % RESEARCH)
    subprocess.run([PY, "-c", code], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated: sec,finra,gdelt,caches")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    steps = [("sec", refresh_sec), ("finra", refresh_finra), ("gdelt", refresh_gdelt), ("caches", rebuild_caches)]
    failed = []
    for name, fn in steps:
        if name in skip:
            log(f"{name}: skipped")
            continue
        log(f"{name}: start")
        try:
            fn()
            log(f"{name}: done")
        except Exception as e:                      # keep going; report at the end
            log(f"{name}: FAILED {e!r}")
            failed.append(name)
    log(f"refresh finished; failed steps: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
