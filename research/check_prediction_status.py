"""Check current prediction data status."""
import pandas as pd
from datetime import datetime

df = pd.read_parquet('data/stocks_with_time_windows.parquet')
latest = df['date'].max()
today = datetime.now()

print("=" * 60)
print("CURRENT PREDICTION DATA STATUS")
print("=" * 60)

print(f"\nToday: {today.date()}")
print(f"Latest data: {latest.date()}")
print(f"Time gap: {(today.date() - latest.date()).days} days")

print(f"\nData contains: {len(df[df['date'] == latest])} stocks")

features = [c for c in df.columns
           if c not in ['date','symbol','open','high','low','close','volume','future_return','label']]
print(f"Features: {len(features)}")

# Check if we have market features with _y suffix
y_features = [f for f in features if '_y' in f]
print(f"Market features (_y): {len(y_features)}")

# Check if we have time window features
time_features = [f for f in features if any(f'{d}d' in f for d in [3,7,15,30])]
print(f"Time window features (3d/7d/15d/30d): {len(time_features)}")

print("\n" + "=" * 60)
print("READY TO PREDICT")
print("=" * 60)
print(f"\nUsing {latest.date()} data to predict:")
print(f"  -> Target: {today.date()} + 1 trading day")
print(f"  -> For Monday Feb 2, 2026")
