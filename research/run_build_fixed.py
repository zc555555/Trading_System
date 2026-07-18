#!/usr/bin/env python
"""Fixed runner for build_dataset with correct paths"""
import sys
from pathlib import Path

# Add features to path
sys.path.insert(0, str(Path(__file__).parent / "features"))

from build_dataset import DatasetBuilder

if __name__ == "__main__":
    builder = DatasetBuilder()

    # Override parquet path to use data/stocks.parquet
    builder.config['data']['parquet_path'] = 'data/stocks.parquet'

    result = builder.build_full_pipeline()
    print("\n=== BUILD COMPLETE ===")
