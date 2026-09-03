"""Pins the GDELT organisation-name normaliser (research/data/gdelt_bigquery.py)
to GDELT's observed spelling: apostrophes and the possessive dropped
("mcdonald", "lowe companies", "oreilly automotive"), corporate suffixes kept
on GDELT's side and stripped from ours, share-class tails removed, hyphen and
ampersand alternatives, ordinary-word cores never matched bare, and nothing
symbol-like."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research" / "data"))

gb = pytest.importorskip("gdelt_bigquery")


def test_possessive_and_apostrophe_forms():
    v = gb._norm_variants(["McDonald's Corporation"])
    assert {"mcdonald", "mcdonalds", "mcdonald corporation"} <= set(v)
    assert not any(" s" in x for x in v)                        # no "mcdonald s"
    assert "oreilly automotive" in gb._norm_variants(["O'Reilly Automotive, Inc."])
    assert "lowe" in gb._norm_variants(["Lowe's Companies, Inc."])   # GDELT "lowe companies" is suffix-stripped in SQL


def test_suffix_and_share_class_handling():
    assert gb._norm_variants(["The Hershey Company"]) == ["hershey", "hershey company"]
    assert gb._norm_variants(["Allstate Corporation (The)"]) == ["allstate", "allstate corporation"]
    assert "booking holdings inc" in gb._norm_variants(["Booking Holdings Inc. Common St"])
    assert "3m" in gb._norm_variants(["3M Company"])               # digit cores may be short
    assert gb._norm_variants(["K"]) == [] and gb._norm_variants(["HES"]) == []


def test_bare_ordinary_words_are_never_matched_alone():
    for name, bare in (("Match Group Inc", "match"), ("Target Corp", "target"), ("Booking Holdings Inc", "booking")):
        v = gb._norm_variants([name])
        assert bare not in v and len(v) >= 1


def test_hyphen_and_ampersand_alternatives():
    v = gb._norm_variants(["Colgate-Palmolive Company"])
    assert {"colgate-palmolive", "colgate palmolive"} <= set(v)
    v = gb._norm_variants(["AT&T Inc."])
    assert {"at&t", "at and t", "at t"} <= set(v)


def test_sql_strips_the_same_suffixes_on_gdelt_side():
    sql = gb.year_sql(2025)
    assert "REGEXP_REPLACE" in sql and "GKGRECORDID" in sql and "SELECT DISTINCT" in sql
    assert gb.SUFFIX_RE in sql
    assert "_PARTITIONTIME >= TIMESTAMP('2025-01-01')" in sql and "< TIMESTAMP('2026-01-01')" in sql
