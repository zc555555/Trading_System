"""Falsifiable factor proposal (AQuA Listing 1, adapted to rulebook track B).

A candidate enters the harness as a proposal, never as a bare expression:
hypothesis, mechanism, the direction it is expected to predict, and the
conditions under which it should be considered refuted, all stated before
any score is seen. `expected_direction` is binding: the track-B gate tests
the holdout one-sided in that direction and fails a candidate whose holdout
sign disagrees (no post-hoc sign calibration).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from mining import dsl

ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,30}$")
DIRECTIONS = ("positive", "negative")
RESERVED_IDS = {"momentum", "trend", "volatility", "volume", "market", "alpha", "high52",
                "pred", "fold", "date", "symbol"}


@dataclass
class Proposal:
    candidate_id: str
    expression: str
    expected_direction: str          # 'positive': higher value -> higher 20d return
    hypothesis: str
    mechanism: str
    refutation_conditions: list[str] = field(default_factory=list)
    mechanism_tag: str = ""          # short, stable mechanism name; matches beliefs.json entries
    source: str = ""                 # e.g. model name / run id
    horizon: int = 20

    def validate(self) -> dict:
        errs = []
        if not ID_RE.match(self.candidate_id or ""):
            errs.append("candidate_id must match ^[a-z][a-z0-9_]{2,30}$")
        if self.candidate_id in RESERVED_IDS:
            errs.append(f"candidate_id '{self.candidate_id}' is reserved")
        if self.expected_direction not in DIRECTIONS:
            errs.append("expected_direction must be 'positive' or 'negative'")
        if len((self.hypothesis or "").strip()) < 20:
            errs.append("hypothesis must be a sentence (>= 20 chars)")
        if len((self.mechanism or "").strip()) < 20:
            errs.append("mechanism must be a sentence (>= 20 chars)")
        if not self.refutation_conditions or not all(
                isinstance(c, str) and c.strip() for c in self.refutation_conditions):
            errs.append("at least one non-empty refutation condition is required")
        if int(self.horizon) not in (5, 20):
            errs.append("track B evaluates horizons 5 and 20 only")
        stats = {}
        try:
            stats = dsl.validate(dsl.parse(self.expression))
        except dsl.DSLError as e:
            errs.append(f"expression: {e}")
        if errs:
            raise ValueError("; ".join(errs))
        return stats

    @property
    def proposal_hash(self) -> str:
        try:
            expr = dsl.canonical(self.expression)
        except dsl.DSLError:          # unparseable: hash the raw text so the record still exists
            expr = f"RAW:{self.expression}"
        key = f"{self.candidate_id}|{expr}|{self.expected_direction}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


def load_proposals(path: str | Path) -> list[Proposal]:
    """Read a JSON file holding one proposal, a list, or {"proposals": [...]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "proposals" in raw:
        raw = raw["proposals"]
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in raw:
        allowed = {k: item[k] for k in Proposal.__dataclass_fields__ if k in item}
        out.append(Proposal(**allowed))
    ids = [p.candidate_id for p in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate_id in proposals file")
    return out
