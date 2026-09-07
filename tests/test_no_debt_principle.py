"""Pins the permanent no-debt risk principle (user directive 2026-08-11).

Losing invested capital is acceptable; owing money is structurally
forbidden. These tests fail CI if anyone weakens the controls without
consciously rewriting both the config block and this file.
"""

import config_trading as cfg


def test_short_caps_exist_and_are_sane():
    # shorts are permitted, but only under hard caps
    assert hasattr(cfg, "SHORT_SINGLE_MAX_PCT")
    assert hasattr(cfg, "SHORT_GROSS_MAX_PCT")
    assert 0 < cfg.SHORT_SINGLE_MAX_PCT <= 5.0, \
        "single-short cap must stay small enough that an extreme gap cannot dent equity"
    assert 0 < cfg.SHORT_GROSS_MAX_PCT <= 50.0
    assert cfg.SHORT_SINGLE_MAX_PCT <= cfg.SHORT_GROSS_MAX_PCT


def test_no_leverage_guard_is_wired():
    """The order path must contain the equity-based exposure block --
    grep-level pin so a refactor cannot silently drop it. Since the 2026-09
    execution rewrite the guard lives in trading/execution.entry_allowed and
    the orchestrator must call it."""
    from pathlib import Path
    root = Path(__file__).parent.parent
    guards = (root / "trading" / "execution.py").read_text(encoding="utf-8")
    orchestrator = (root / "run_staggered_trading.py").read_text(encoding="utf-8")
    assert "no leverage, ever" in guards
    assert "entry_allowed" in orchestrator
    assert "SHORT_SINGLE_MAX_PCT" in orchestrator
    assert "SHORT_GROSS_MAX_PCT" in orchestrator


def test_principle_block_present():
    from pathlib import Path
    src = (Path(__file__).parent.parent / "config_trading.py") \
        .read_text(encoding="utf-8")
    assert "NO DEBT, EVER" in src
    assert "NO LEVERAGE" in src
