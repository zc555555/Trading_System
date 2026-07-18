"""Quick script to check the status of parquet files."""
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "data"

files = ['stocks.parquet', 'stocks_selected_features.parquet', 'stocks_with_time_windows.parquet']

for filename in files:
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"\n{filename}: NOT FOUND")
        continue

    df = pd.read_parquet(filepath)
    print(f"\n{filename}:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Shape: {df.shape}")

    # Check for _y suffix
    y_cols = [c for c in df.columns if '_y' in c]
    print(f"  Has _y suffix: {len(y_cols) > 0} ({len(y_cols)} columns)")
    if y_cols:
        print(f"    Examples: {y_cols[:5]}")

    # Check for time window features (3d, 7d, 15d, 30d)
    time_window_cols = [c for c in df.columns
                        if any(f'_{d}d' in c for d in [3, 7, 15, 30])]
    print(f"  Has time windows (3d/7d/15d/30d): {len(time_window_cols) > 0} ({len(time_window_cols)} columns)")
    if time_window_cols:
        print(f"    Examples: {time_window_cols[:5]}")
