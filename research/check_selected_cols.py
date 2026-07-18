"""Check columns in stocks_selected_features.parquet."""
import pandas as pd
from pathlib import Path

df = pd.read_parquet(Path(__file__).parent / "data" / "stocks_selected_features.parquet")

print("Columns in stocks_selected_features.parquet:")
for i, c in enumerate(df.columns):
    print(f"{i+1:2d}. {c}")

print(f"\nTotal columns: {len(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
