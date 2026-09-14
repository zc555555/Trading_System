"""Registry of agent-mined factors adopted under rulebook track B.

`mined_factors.json` is the frozen list of expressions that passed the
track-B gate and were adopted by a human (mining/harness.py adopt). The
production feature path reads it in four places so that adoption is one
registry entry plus a retrain, with no hand edits:

    features/build_dataset.py      computes `mined_<id>` from the expression
    features/cross_sectional.py    adds `mined_<id>` to the _xs list
    update_selected_features.py    keeps the column through feature selection
    factors/factor_definitions.py  registers a factor group per adopted id

An empty registry leaves every one of those unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REGISTRY = Path(__file__).resolve().with_name("mined_factors.json")


PRODUCTION_HORIZON = 20
PRODUCTION_UNIVERSE = "sp500"     # the live book trades S&P members; mid-cap is research only (RULEBOOK "Universes")


def load_adopted(path: Path = REGISTRY, production_only: bool = True) -> list[dict]:
    """Adopted entries. With production_only (the default, used by every
    production consumer) entries validated at another horizon or on another
    universe -- the "shelf" -- are excluded, so they can never reach the
    live book."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = list(data.get("adopted", []))
    if production_only:
        entries = [e for e in entries if int(e.get("horizon", PRODUCTION_HORIZON)) == PRODUCTION_HORIZON
                   and str(e.get("universe", PRODUCTION_UNIVERSE)) == PRODUCTION_UNIVERSE
                   and e.get("status", "production") != "shelf"]
    return entries


def compile_entry(entry: dict, work: pd.DataFrame):
    """The production feature of one adopted entry on a panel that already
    carries the auxiliary fields: a single expression compiles with the DSL;
    a pool release ('type': 'pool') is the equal-weight composite of the
    signed per-date ranks of its FROZEN members (mining/pool.py), member by
    member, so a member whose source is missing raises instead of being
    silently zero-filled."""
    from mining import dsl
    if entry.get("type") == "pool":
        from mining import pool as pl
        dates = pd.to_datetime(work["date"]).to_numpy()
        cols = {}
        for m in entry["members"]:
            feat = dsl.compile_expression(m["expression"], work).to_numpy(dtype=float)
            cols[m["hash"]] = pl.signed_rank(dates, feat, 1.0 if m["expected_direction"] == "positive" else -1.0)
        return pl.composite(pd.DataFrame(cols, index=work.index))
    return dsl.compile_expression(entry["expression"], work).to_numpy()


def feature_column(entry: dict) -> str:
    return f"mined_{entry['id']}"


def adopted_feature_columns(path: Path = REGISTRY) -> list[str]:
    return [feature_column(e) for e in load_adopted(path)]


def mined_factor_groups(path: Path = REGISTRY) -> dict[str, list[str]]:
    return {e["id"]: [feature_column(e), feature_column(e) + "_xs"] for e in load_adopted(path)}


def add_adopted_mined_features(df: pd.DataFrame, path: Path = REGISTRY) -> pd.DataFrame:
    """Compile every adopted expression onto the OHLCV panel with the same
    DSL the harness evaluated it with."""
    entries = load_adopted(path)
    if not entries:
        return df
    research = Path(__file__).resolve().parent.parent
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    from mining import aux_fields
    df = df.copy()
    work = aux_fields.attach(df)                # Sharadar fields when the source files exist
    for e in entries:
        # a missing source raises here (DSLError) rather than silently zero-filling a live feature
        df[feature_column(e)] = compile_entry(e, work)
    return df


def register_adopted(entry: dict, path: Path = REGISTRY) -> None:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    adopted = list(data.get("adopted", []))
    if any(e["id"] == entry["id"] for e in adopted):
        raise ValueError(f"{entry['id']} is already in the registry")
    adopted.append(entry)
    data["adopted"] = adopted
    data.setdefault("_doc", "Agent-mined factors adopted under rulebook track B. "
                            "Edit only via mining/harness.py adopt.")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
