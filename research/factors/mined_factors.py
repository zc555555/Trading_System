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


def load_adopted(path: Path = REGISTRY) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("adopted", []))


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
    from mining import dsl, aux_fields
    df = df.copy()
    work = aux_fields.attach(df)                # Sharadar fields when the source files exist
    for e in entries:
        # a missing source raises here (DSLError) rather than silently zero-filling a live feature
        df[feature_column(e)] = dsl.compile_expression(e["expression"], work).to_numpy()
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
