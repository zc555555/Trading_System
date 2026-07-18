"""
快速检查数据是否最新
"""
import pandas as pd
from datetime import datetime, timedelta

# 读取数据
df = pd.read_parquet('data/stocks.parquet')

# 最新日期
latest_date = df['date'].max()
latest_date_naive = latest_date.replace(tzinfo=None)  # Remove timezone
latest_date_str = latest_date.strftime('%Y-%m-%d')

# 今天日期
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')

# 计算差距（工作日）
days_behind = (today - latest_date_naive).days

print("=" * 70)
print("DATA FRESHNESS CHECK")
print("=" * 70)
print(f"Today:              {today_str} ({today.strftime('%A')})")
print(f"Latest data date:   {latest_date_str} ({latest_date.strftime('%A')})")
print(f"Days behind:        {days_behind} days")
print()

# 判断是否需要刷新
if days_behind == 0:
    print("[STATUS] ✅ Data is UP TO DATE")
elif days_behind == 1 and today.weekday() == 0:  # Monday
    print("[STATUS] ✅ OK (Today is Monday, latest is Friday)")
elif days_behind <= 3 and today.weekday() in [5, 6]:  # Weekend
    print("[STATUS] ✅ OK (Weekend, waiting for next trading day)")
else:
    print(f"[STATUS] ⚠️  DATA IS OUTDATED by {days_behind} days")
    print(f"[ACTION] Run: python data/fetch_ohlcv.py")

print()
print("Last 5 trading days in data:")
for date in sorted(df['date'].unique())[-5:]:
    print(f"  - {date.strftime('%Y-%m-%d %A')}")
print("=" * 70)
