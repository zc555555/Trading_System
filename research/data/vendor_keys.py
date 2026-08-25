"""Resolve vendor API keys: environment variable first, then the git-ignored
``config_keys.py`` at the repo root. Returns '' when neither is set so
callers can fail loudly with a useful message."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def get_key(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    cfg = ROOT / "config_keys.py"
    if cfg.exists():
        spec = importlib.util.spec_from_file_location("config_keys", cfg)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return str(getattr(mod, name, "") or "").strip()
    return ""
