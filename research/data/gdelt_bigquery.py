"""GDELT daily tone/volume via the BigQuery public dataset (no API throttle).

The GDELT DOC API rate-limits aggressively; the same data lives in the
`gdelt-bq.gdeltv2.gkg_partitioned` public table. This module:

1. generates per-year SQL that maps GKG organization mentions to our
   symbols (using the cached company_names.json variants) and aggregates
   daily average tone + article counts,
2. runs the queries with google-cloud-bigquery (route B: local gcloud
   auth), or emits .sql files for manual console use (route A),
3. converts results into data/gdelt_daily.parquet with the exact schema
   fetch_gdelt_timeline.py produces ([date, symbol, gdelt_tone,
   gdelt_articles]), so evaluation/experiments/news_factor.py runs
   unchanged.

Cost note: per-year queries prune partitions; scanning V2Organizations +
V2Tone for 2017-2026 totals roughly 0.6-0.9TB -- inside the free 1TB
sandbox month, but run the --dry-run first and don't repeat full years.

Usage:
    python data/gdelt_bigquery.py --emit-sql          # route A: write .sql files
    python data/gdelt_bigquery.py --dry-run           # route B: cost estimate
    python data/gdelt_bigquery.py --years 2017 2018   # route B: fetch years
    python data/gdelt_bigquery.py                     # route B: all years
    python data/gdelt_bigquery.py --ingest-csv dir/   # route A: fold in console exports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent
NAMES_CACHE = DATA_DIR / "company_names.json"
OUTPUT = DATA_DIR / "gdelt_daily.parquet"
SQL_DIR = DATA_DIR / "gdelt_sql"
YEARS = list(range(2017, 2027))

GENERIC = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co",
           "co.", "ltd", "ltd.", "plc", "group", "holdings", "class", "the",
           "&", "and"}


def _norm_variants(variants: list[str]) -> list[str]:
    """Lowercased org-name variants incl. suffix-stripped forms, as GDELT
    normalizes organization names to lowercase without punctuation.

    Punctuation must be removed BEFORE tokenizing: yfinance longNames like
    "Salesforce, Inc." otherwise leave a "salesforce," token that never
    matches GDELT's "salesforce".
    """
    out = set()
    for v in variants:
        v = re.sub(r'["().,]', "", v).strip().lower()
        for candidate in (v, v.replace("&", " ")):
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) >= 3:
                out.add(candidate)
            words = [w for w in candidate.split() if w not in GENERIC]
            stripped = " ".join(words)
            if len(stripped) >= 4:
                out.add(stripped)
    return sorted(out)


def build_name_rows() -> list[tuple[str, str]]:
    cache = json.loads(NAMES_CACHE.read_text(encoding="utf-8"))
    rows = []
    for symbol, variants in cache.items():
        for name in _norm_variants(variants):
            rows.append((symbol, name.replace("'", "\\'")))
    return rows


def year_sql(year: int) -> str:
    rows = build_name_rows()
    structs = ",\n    ".join(
        f"STRUCT('{sym}' AS symbol, '{name}' AS org)" for sym, name in rows)
    return f"""-- GDELT daily tone/volume for {year} (partition-pruned)
WITH name_map AS (
  SELECT * FROM UNNEST([
    {structs}
  ])
)
SELECT
  DATE(_PARTITIONTIME) AS date,
  nm.symbol AS symbol,
  AVG(CAST(SPLIT(g.V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS gdelt_tone,
  COUNT(*) AS gdelt_articles
FROM `gdelt-bq.gdeltv2.gkg_partitioned` AS g,
  UNNEST(SPLIT(g.V2Organizations, ';')) AS org_entry
JOIN name_map AS nm
  ON LOWER(SPLIT(org_entry, ',')[OFFSET(0)]) = nm.org
WHERE _PARTITIONTIME >= TIMESTAMP('{year}-01-01')
  AND _PARTITIONTIME < TIMESTAMP('{year + 1}-01-01')
  AND g.V2Tone IS NOT NULL
  AND g.V2Organizations IS NOT NULL
GROUP BY date, symbol
"""


def emit_sql_files():
    SQL_DIR.mkdir(exist_ok=True)
    for year in YEARS:
        (SQL_DIR / f"gdelt_{year}.sql").write_text(year_sql(year),
                                                   encoding="utf-8")
    print(f"wrote {len(YEARS)} files -> {SQL_DIR}")
    print("Run each in the BigQuery console, export results as CSV, then:")
    print("  python data/gdelt_bigquery.py --ingest-csv <download_dir>")


def _merge_and_save(frames: list[pd.DataFrame]):
    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    # cast away BigQuery's nullable extension dtypes (Float64Dtype etc.)
    df['gdelt_tone'] = pd.to_numeric(df['gdelt_tone'], errors='coerce').astype('float64')
    df['gdelt_articles'] = pd.to_numeric(df['gdelt_articles'], errors='coerce').astype('float64')
    df = (df.groupby(['date', 'symbol'], as_index=False)
            .agg(gdelt_tone=('gdelt_tone', 'mean'),
                 gdelt_articles=('gdelt_articles', 'sum'))
            .sort_values(['symbol', 'date']))
    df.to_parquet(OUTPUT, index=False)
    print(f"saved: {df['symbol'].nunique()} symbols, {len(df):,} rows "
          f"({df['date'].min().date()} .. {df['date'].max().date()}) -> {OUTPUT.name}")


def ingest_csv(folder: Path):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {folder}")
    frames = [pd.read_csv(f) for f in files]
    print(f"ingesting {len(files)} csv files...")
    _merge_and_save(frames)


def run_queries(years: list[int], dry_run: bool,
                project: str | None = None):
    import os
    from google.cloud import bigquery
    client = bigquery.Client(
        project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    frames = []
    if OUTPUT.exists() and not dry_run:
        existing = pd.read_parquet(OUTPUT)
        done_years = set(pd.to_datetime(existing['date']).dt.year.unique())
        keep = existing[~pd.to_datetime(existing['date']).dt.year.isin(
            [y for y in years])]
        frames.append(keep)
        print(f"existing parquet: keeping years outside requested set")
    total_bytes = 0
    for year in years:
        sql = year_sql(year)
        cfg = bigquery.QueryJobConfig(dry_run=dry_run,
                                      use_query_cache=True)
        job = client.query(sql, job_config=cfg)
        if dry_run:
            gb = job.total_bytes_processed / 1e9
            total_bytes += job.total_bytes_processed
            print(f"  {year}: would scan {gb:,.1f} GB")
            continue
        df = job.result().to_dataframe()
        print(f"  {year}: {len(df):,} rows, "
              f"{job.total_bytes_processed / 1e9:,.1f} GB scanned")
        total_bytes += job.total_bytes_processed or 0
        frames.append(df)
    print(f"total scanned: {total_bytes / 1e9:,.1f} GB")
    if not dry_run and frames:
        _merge_and_save(frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-sql", action="store_true")
    ap.add_argument("--ingest-csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--project", type=str, default="stockpredict-503112")
    args = ap.parse_args()

    if args.emit_sql:
        emit_sql_files()
    elif args.ingest_csv:
        ingest_csv(args.ingest_csv)
    else:
        run_queries(args.years or YEARS, dry_run=args.dry_run,
                    project=args.project)
