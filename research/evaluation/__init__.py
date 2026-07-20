"""Trustworthy evaluation layer (P1, 2026-07).

Replaces the single-split pooled-Pearson methodology with:
- per-date cross-sectional rank IC (metrics.py)
- purged walk-forward with embargo (purged_walk_forward.py)
- cost-aware next-day-execution portfolio simulation (simulate_portfolio.py)
- a shared factor-ensemble training core (factor_training.py) used by both
  the walk-forward and the production trainer, so the two can never drift.
"""
