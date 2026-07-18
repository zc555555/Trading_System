"""
Centralized config loader.

Loads research/config.yaml and, if ``data.symbols`` is null/empty, resolves
the universe from ``data.universe`` (and optionally appends benchmark ETFs).
The resolved list is written back into the dict so that downstream code
which reads ``config['data']['symbols']`` keeps working unchanged.

This is the W1 (2026-05) migration glue between the old "explicit symbol list
in YAML" world and the new "named universe" world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from data.universe import get_universe


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Optional[Path] = None, resolve_symbols: bool = True) -> Dict[str, Any]:
    """Load research/config.yaml and (optionally) resolve the universe.

    Args:
        config_path: Override path to config.yaml. Defaults to the file
            colocated with this module.
        resolve_symbols: If True (default), populate ``data.symbols`` from
            ``data.universe`` when it is null/empty. Set False to inspect
            the raw config.

    Returns:
        Parsed config dict. When ``resolve_symbols`` is True the
        ``data.symbols`` field is guaranteed to be a non-empty list.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not resolve_symbols:
        return config

    data_cfg = config.setdefault("data", {})
    explicit = data_cfg.get("symbols")
    if explicit:
        # legacy explicit list — keep as-is, signals an intentional override
        return config

    universe_name = data_cfg.get("universe", "sp100")
    include_benchmarks = bool(data_cfg.get("include_benchmark_etfs", False))
    data_cfg["symbols"] = get_universe(universe_name, include_benchmarks=include_benchmarks)
    return config


if __name__ == "__main__":
    cfg = load_config()
    symbols = cfg["data"]["symbols"]
    print(f"Resolved universe: {len(symbols)} tickers")
    print(f"First 10: {symbols[:10]}")
