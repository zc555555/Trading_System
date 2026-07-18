"""Check if we have Jan 30 data available."""
import yfinance as yf
from datetime import datetime, timedelta

print("=" * 80)
print("Checking Yahoo Finance for Jan 30, 2026 data")
print("=" * 80)

# Test with AAPL
ticker = yf.Ticker('AAPL')
today = datetime.now()
end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')

df = ticker.history(start='2026-01-25', end=end_date)

print("\nAAPL recent data:")
print(df[['Close', 'Volume']].tail(10))

last_date = df.index[-1].date()
print(f"\nLast trading day available: {last_date}")

if str(last_date) == '2026-01-30':
    print("[OK] Jan 30 data is available!")
elif str(last_date) == '2026-01-29':
    print("[INFO] Last available is Jan 29 (market might not have Jan 30 data yet)")
else:
    print(f"[INFO] Last available is {last_date}")
