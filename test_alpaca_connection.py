"""
TEST ALPACA CONNECTION

Quick test to verify your Alpaca API connection works.
Run this first before attempting auto-trading.
"""

import sys
from datetime import datetime

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
except ImportError:
    print("\n[ERROR] Alpaca Python library not installed!")
    print("\nPlease install it:")
    print("  pip install alpaca-py")
    print()
    sys.exit(1)

from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


def test_connection():
    """Test Alpaca API connection"""
    print("\n" + "="*80)
    print("ALPACA CONNECTION TEST")
    print("="*80)
    print(f"Mode: {'PAPER TRADING' if ALPACA_PAPER else 'LIVE TRADING'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test 1: Trading Client
    print("[Test 1/3] Testing Trading Client...")
    try:
        trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER
        )

        account = trading_client.get_account()
        print(f"[OK] Connected successfully!")
        print(f"  Account ID: {account.account_number}")
        print(f"  Status: {account.status}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print()
    except Exception as e:
        print(f"[FAILED] Trading client error: {e}")
        print()
        return False

    # Test 2: Market Data Client
    print("[Test 2/3] Testing Market Data Client...")
    try:
        data_client = StockHistoricalDataClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY
        )

        # Get a quote
        request = StockLatestQuoteRequest(symbol_or_symbols="AAPL")
        quote = data_client.get_stock_latest_quote(request)

        aapl_price = float(quote["AAPL"].ask_price)
        print(f"[OK] Market data access successful!")
        print(f"  AAPL Current Price: ${aapl_price:.2f}")
        print()
    except Exception as e:
        print(f"[FAILED] Market data error: {e}")
        print()
        return False

    # Test 3: Get Positions
    print("[Test 3/3] Testing Position Query...")
    try:
        positions = trading_client.get_all_positions()
        print(f"[OK] Position query successful!")
        print(f"  Current Positions: {len(positions)}")

        if len(positions) > 0:
            print(f"\n  Open Positions:")
            for pos in positions:
                pl_sign = '+' if float(pos.unrealized_pl) >= 0 else ''
                print(f"    {pos.symbol}: {pos.qty} shares, "
                      f"P&L: {pl_sign}${float(pos.unrealized_pl):.2f}")
        print()
    except Exception as e:
        print(f"[FAILED] Position query error: {e}")
        print()
        return False

    # All tests passed
    print("="*80)
    print("[SUCCESS] All tests passed! Your Alpaca connection is working.")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run: python alpaca_trader.py")
    print("  2. Or run: auto_trading.bat")
    print()

    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
