"""Pins the sector-neutral operators (dsl.sector_rank / sector_demean) and
the sector column (aux_fields.attach_sector): ranks are taken within
(date, sector), missing sectors are 'Unknown', and the operators refuse a
panel without the column."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402


def _panel():
    dates = pd.bdate_range("2024-01-01", periods=3, tz="America/New_York")
    rows = []
    for d in dates:
        for i, s in enumerate(("A1", "A2", "A3", "B1", "B2", "C1")):
            rows.append({"date": d, "symbol": s, "close": float(i + 1), "open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0})
    return pd.DataFrame(rows)


def test_sector_rank_is_within_sector_and_unknown_is_a_bucket():
    df = ax.attach_sector(_panel(), table={"A1": "Tech", "A2": "Tech", "A3": "Tech", "B1": "Energy", "B2": "Energy"})
    assert set(df["sector"]) == {"Tech", "Energy", "Unknown"}
    r = dsl.compile_expression("sector_rank(close)", df)
    day = df[df["date"] == df["date"].iloc[0]].assign(r=r[df["date"] == df["date"].iloc[0]].to_numpy())
    by = day.set_index("symbol")["r"]
    assert abs(by["A1"] - 1 / 3) < 1e-12 and abs(by["A3"] - 1.0) < 1e-12        # ranks inside Tech
    assert abs(by["B1"] - 0.5) < 1e-12 and abs(by["B2"] - 1.0) < 1e-12          # inside Energy
    assert abs(by["C1"] - 1.0) < 1e-12                                          # alone in Unknown
    d = dsl.compile_expression("sector_demean(close)", df)
    assert abs(d[df["symbol"] == "A2"].iloc[0] - 0.0) < 1e-12 and abs(d[df["symbol"] == "C1"].iloc[0]) < 1e-12
    assert "sector_rank" in dsl.describe_ops()


def test_sector_ops_refuse_a_panel_without_the_column():
    try:
        dsl.compile_expression("sector_rank(close)", _panel())
    except dsl.DSLError as e:
        assert "sector" in str(e)
    else:
        raise AssertionError("expected a DSLError")
