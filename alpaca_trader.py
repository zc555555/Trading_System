"""
ALPACA AUTO TRADER - Multi-Factor Stock Trading System

Integrates with your multi-factor model to automatically execute trades on Alpaca.

Features:
- Automatic trade execution based on multi-factor signals
- Risk management (stop-loss, position limits)
- Daily performance tracking
- Paper Trading support
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
except ImportError:
    print("[ERROR] Alpaca API not installed!")
    print("Run: pip install alpaca-py")
    exit(1)

from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


class AlpacaAutoTrader:
    """Automated trading system using Alpaca API"""

    def __init__(self):
        """Initialize Alpaca trading client"""
        self.paper = ALPACA_PAPER

        print(f"\n{'='*80}")
        print(f"ALPACA AUTO TRADER - {'PAPER TRADING' if self.paper else 'LIVE TRADING'}")
        print(f"{'='*80}\n")

        # Initialize clients
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=self.paper
        )

        self.data_client = StockHistoricalDataClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY
        )

        # Trading parameters
        self.max_position_pct = 0.15  # Max 15% per stock
        self.stop_loss_pct = 0.025   # 2.5% stop loss
        self.max_daily_loss_pct = 0.03  # 3% max daily loss

        # Paths
        self.project_dir = Path(__file__).parent
        self.artifacts_dir = self.project_dir / "research" / "artifacts"
        self.logs_dir = self.project_dir / "trading_logs"
        self.logs_dir.mkdir(exist_ok=True)

    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'initial_equity': float(account.last_equity),
                # daytrade_count can come back None on paper accounts
                'day_trade_count': int(account.daytrade_count or 0)
            }
        except Exception as e:
            print(f"[ERROR] Failed to get account info: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc) * 100,
                    'side': pos.side
                })
            return result
        except Exception as e:
            # UNKNOWN is not "flat" (2026-09 review): callers must not treat a
            # failed query as an empty book
            raise RuntimeError(f"positions unavailable: {e}") from e

    def close_all_positions(self):
        """Close all open positions (for 1-day holding strategy)"""
        try:
            positions = self.get_positions()
        except RuntimeError as e:
            print(f"[ABORT] {e} -- not closing, not treating the account as flat")
            return False

        if len(positions) == 0:
            print("[INFO] No positions to close")
            return True

        print(f"\n[INFO] Closing {len(positions)} positions...")
        print("-" * 80)

        for pos in positions:
            try:
                self.trading_client.close_position(pos['symbol'])
                pl_sign = '+' if pos['unrealized_pl'] >= 0 else ''
                print(f"[SELL] {pos['symbol']:<6} {pos['qty']:<8.0f} shares  "
                      f"P&L: {pl_sign}${pos['unrealized_pl']:>8.2f} ({pl_sign}{pos['unrealized_plpc']:.2f}%)")
            except Exception as e:
                print(f"[ERROR] Failed to close {pos['symbol']}: {e}")

        print("-" * 80)

    def load_signals(self, signal_file: Optional[Path] = None) -> Dict:
        """Load trading signals from multi-factor model"""
        if signal_file is None:
            # Find latest signal file
            today = datetime.now().strftime('%Y%m%d')
            signal_file = self.artifacts_dir / f"signals_multi_factor_{today}.json"

            if not signal_file.exists():
                # Try yesterday
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                signal_file = self.artifacts_dir / f"signals_multi_factor_{yesterday}.json"

        if not signal_file.exists():
            print(f"[ERROR] Signal file not found: {signal_file}")
            return None

        with open(signal_file, 'r') as f:
            signals = json.load(f)

        print(f"[OK] Loaded signals from: {signal_file.name}")
        return signals

    def get_current_price(self, symbol: str) -> float:
        """Current price from Alpaca's latest trade (same venue we execute
        on), falling back to Yahoo Finance if the data API fails.

        P2 (2026-07): previously yfinance-only, which meant sizing and P&L
        used a different price source than execution.
        """
        try:
            trade = self.data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol))
            price = float(trade[symbol].price)
            if price > 0:
                return price
        except Exception as e:
            print(f"[WARNING] Alpaca price failed for {symbol} ({e}), trying yfinance")

        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period='1d')
            if hist.empty:
                print(f"[WARNING] No price data for {symbol}")
                return None
            price = float(hist['Close'].iloc[-1])
            if price <= 0:
                print(f"[WARNING] Invalid price for {symbol}: {price}")
                return None
            return price
        except Exception as e:
            print(f"[WARNING] Failed to get price for {symbol}: {e}")
            return None

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol (e.g. resting bracket legs).

        Must be called before closing a position whose quantity is tied up
        in bracket child orders -- Alpaca rejects the close otherwise.
        """
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN,
                                 symbols=[symbol]))
        except Exception as e:
            print(f"[WARNING] Could not list open orders for {symbol}: {e}")
            return 0
        n = 0
        for order in open_orders:
            try:
                self.trading_client.cancel_order_by_id(order.id)
                n += 1
            except Exception as e:
                print(f"[WARNING] Cancel failed for {symbol} order {order.id}: {e}")
        if n:
            print(f"[OK] Canceled {n} open order(s) for {symbol}")
            # Cancellation is asynchronous at the broker: the qty stays
            # held_for_orders until the cancel is processed, and a close
            # submitted in that window is rejected with 40310000
            # "insufficient qty available" (2026-08-26: six scheduled
            # closes failed this way). Wait until nothing is open.
            self.wait_until_no_open_orders(symbol)
        return n

    def wait_until_no_open_orders(self, symbol: str, timeout_s: float = 15.0,
                                  poll_s: float = 1.0) -> bool:
        """Block until the broker reports no open orders for ``symbol``
        (cancellations processed). Returns False on timeout."""
        import time as _time
        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            try:
                still = self.trading_client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
            except Exception as e:
                print(f"[WARNING] Could not poll open orders for {symbol}: {e}")
                return False
            if not still:
                return True
            _time.sleep(poll_s)
        print(f"[WARNING] {symbol}: open orders still pending cancel after {timeout_s:.0f}s")
        return False

    def place_bracket_order(self, symbol: str, qty: int, side: str,
                            stop_price: float, take_price: float,
                            client_order_id: str | None = None) -> bool:
        """Market entry with broker-side GTC stop-loss and take-profit legs.

        P2 (2026-07): stops previously existed only in the 5-minute monitor
        poller, leaving multi-day positions unprotected overnight. Bracket
        legs live at the broker and survive process crashes and weekends.
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_price, 2)),
                client_order_id=client_order_id,
            )
            order = self.trading_client.submit_order(request)

            order_id = getattr(order, 'id', None)
            if order_id is None:
                print(f"[ERROR] Bracket order for {symbol}: no order id returned")
                return False

            bad_statuses = {'rejected', 'canceled', 'expired', 'suspended'}
            for _ in range(5):
                refreshed = self.trading_client.get_order_by_id(order_id)
                status = str(getattr(refreshed, 'status', '')).lower().split('.')[-1]
                if status in bad_statuses:
                    reason = getattr(refreshed, 'reject_reason', None) or status
                    print(f"[ERROR] Bracket {symbol} {status}: {reason}")
                    return False
                if status in ('accepted', 'new', 'partially_filled', 'filled',
                              'pending_new', 'held'):
                    return True
                time.sleep(1)
            return True
        except Exception as e:
            print(f"[ERROR] Bracket order failed for {symbol}: {e}")
            return False

    def place_limit_bracket_order(self, symbol: str, qty: int, side: str,
                                  limit_price: float, stop_price: float,
                                  take_price: float,
                                  client_order_id: str | None = None) -> bool:
        """LIMIT entry with GTC stop/take children (execution A/B, arm B).

        Posts at the arrival price instead of paying the spread/gap at the
        open. Non-fills are converted to market by the monitor after
        config_trading.LIMIT_TIMEOUT_MIN minutes of regular trading.
        """
        try:
            from alpaca.trading.requests import LimitOrderRequest
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=round(limit_price, 2),
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_price, 2)),
                client_order_id=client_order_id,
            )
            order = self.trading_client.submit_order(request)

            order_id = getattr(order, 'id', None)
            if order_id is None:
                print(f"[ERROR] Limit bracket for {symbol}: no order id returned")
                return False

            bad_statuses = {'rejected', 'canceled', 'expired', 'suspended'}
            for _ in range(5):
                refreshed = self.trading_client.get_order_by_id(order_id)
                status = str(getattr(refreshed, 'status', '')).lower().split('.')[-1]
                if status in bad_statuses:
                    reason = getattr(refreshed, 'reject_reason', None) or status
                    print(f"[ERROR] Limit bracket {symbol} {status}: {reason}")
                    return False
                if status in ('accepted', 'new', 'partially_filled', 'filled',
                              'pending_new', 'held'):
                    return True
                time.sleep(1)
            return True
        except Exception as e:
            print(f"[ERROR] Limit bracket failed for {symbol}: {e}")
            return False

    def place_market_order(self, symbol: str, qty: int, side: str = 'buy') -> bool:
        """Place a market order and confirm it was accepted (not rejected/canceled)."""
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(request)

            # submit_order 返回成功不代表订单被接受，可能 rejected / canceled / expired
            # 轮询最多 5 秒确认订单终态
            order_id = getattr(order, 'id', None)
            if order_id is None:
                print(f"[ERROR] Order for {symbol}: no order id returned")
                return False

            bad_statuses = {'rejected', 'canceled', 'expired', 'suspended'}
            for _ in range(5):
                refreshed = self.trading_client.get_order_by_id(order_id)
                status = str(getattr(refreshed, 'status', '')).lower().split('.')[-1]
                if status in bad_statuses:
                    reason = getattr(refreshed, 'reject_reason', None) or status
                    print(f"[ERROR] Order {symbol} {status}: {reason}")
                    return False
                if status in ('accepted', 'new', 'partially_filled', 'filled', 'pending_new'):
                    return True
                time.sleep(1)
            # 5 秒仍未到达终态，按已提交处理，Alpaca 常见路径
            return True
        except Exception as e:
            print(f"[ERROR] Order failed for {symbol}: {e}")
            return False

    def execute_trades(self, signals: Dict):
        """Execute trades based on multi-factor signals"""
        if not signals['should_trade']:
            print(f"\n[SKIP] No trading signals today ({signals['n_stocks']} stocks < minimum)")
            return

        account = self.get_account_info()
        if not account:
            print("[ERROR] Cannot get account info, aborting")
            return

        capital = account['cash']

        print(f"\n{'='*80}")
        print(f"EXECUTING TRADES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Available Cash: ${capital:,.2f}")
        print(f"Stocks to Trade: {signals['n_stocks']}")
        print()

        # 1. Close all existing positions (1-day holding strategy)
        if self.close_all_positions() is False:
            print("[ABORT] positions unknown -- no new orders")
            return

        # Wait a moment for positions to close
        time.sleep(2)

        # 2. Open new positions based on signals
        print(f"\n{'='*80}")
        print("OPENING NEW POSITIONS")
        print(f"{'='*80}\n")

        trades_executed = []

        for stock in signals['stocks']:
            symbol = stock['symbol']

            # Direction lives in the sign of `prediction` (see
            # get_daily_signals_multi_factor.py). This legacy path is
            # long-only: SELL signals are skipped, never bought. Shorting
            # is handled exclusively by run_staggered_trading.py.
            prediction = float(stock.get('prediction', 0.0))
            if prediction <= 0:
                print(f"[SKIP] {symbol}: SELL signal (prediction {prediction*100:+.2f}%), legacy path is long-only")
                continue

            # Cap any single name at max_position_pct of capital.
            position_pct = min(stock['position_pct'] / 100, self.max_position_pct)
            amount = capital * position_pct

            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price is None:
                print(f"[SKIP] {symbol}: Cannot get price")
                continue

            # Calculate quantity
            qty = int(amount / current_price)

            if qty < 1:
                print(f"[SKIP] {symbol}: Quantity too small ({amount:.2f} / {current_price:.2f})")
                continue

            # Place order
            print(f"[BUY]  {symbol:<6} {qty:>4} shares @ ${current_price:>7.2f} = ${amount:>8.2f}")

            if self.place_market_order(symbol, qty, 'buy'):
                trades_executed.append({
                    'symbol': symbol,
                    'qty': qty,
                    'price': current_price,
                    'amount': amount,
                    'position_pct': position_pct * 100,
                    'timestamp': datetime.now().isoformat()
                })

                # Note: Stop-loss orders in Alpaca Paper Trading might not work perfectly
                # In live trading, you'd set stop-loss here

        print(f"\n{'='*80}")
        print(f"TRADES COMPLETE: {len(trades_executed)} orders executed")
        print(f"{'='*80}\n")

        # Log trades
        self.log_trades(trades_executed, signals)

        return trades_executed

    def log_trades(self, trades: List[Dict], signals: Dict):
        """Log trades to file"""
        log_file = self.logs_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'paper_trading': self.paper,
            'signals': signals,
            'trades_executed': trades,
            'account_snapshot': self.get_account_info()
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"[OK] Trade log saved: {log_file.name}")

    def print_portfolio_status(self):
        """Print current portfolio status"""
        account = self.get_account_info()
        try:
            positions = self.get_positions()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        if not account:
            print("[ERROR] Cannot get account info")
            return

        print(f"\n{'='*80}")
        print(f"PORTFOLIO STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Account Mode:     {'PAPER TRADING' if self.paper else 'LIVE TRADING'}")
        print(f"Total Equity:     ${account['equity']:>12,.2f}")
        print(f"Cash:             ${account['cash']:>12,.2f}")
        print(f"Positions Value:  ${account['equity'] - account['cash']:>12,.2f}")

        # Calculate daily P&L
        if account['initial_equity'] > 0:
            daily_pl = account['equity'] - account['initial_equity']
            daily_pl_pct = (daily_pl / account['initial_equity']) * 100
            pl_sign = '+' if daily_pl >= 0 else ''
            print(f"Daily P&L:        {pl_sign}${daily_pl:>11,.2f} ({pl_sign}{daily_pl_pct:.2f}%)")

        print()

        if len(positions) > 0:
            print(f"{'Symbol':<8} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L $':<12} {'P&L %':<8}")
            print("-" * 70)

            for pos in positions:
                pl_sign = '+' if pos['unrealized_pl'] >= 0 else ''
                print(f"{pos['symbol']:<8} {pos['qty']:<8.0f} "
                      f"${pos['avg_entry_price']:<9.2f} ${pos['current_price']:<9.2f} "
                      f"{pl_sign}${pos['unrealized_pl']:<10.2f} {pl_sign}{pos['unrealized_plpc']:.2f}%")
        else:
            print("No open positions")

        print(f"{'='*80}\n")

    def generate_daily_report(self):
        """Generate end-of-day report"""
        report_file = self.logs_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"

        account = self.get_account_info()
        try:
            positions = self.get_positions()
            positions_note = ""
        except RuntimeError as e:
            positions, positions_note = [], f" (UNAVAILABLE: {e})"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DAILY TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Account Mode: {'PAPER TRADING' if self.paper else 'LIVE TRADING'}\n")
            f.write(f"Total Equity: ${account['equity']:,.2f}\n")
            f.write(f"Cash: ${account['cash']:,.2f}\n\n")

            if account['initial_equity'] > 0:
                daily_pl = account['equity'] - account['initial_equity']
                daily_pl_pct = (daily_pl / account['initial_equity']) * 100
                f.write(f"Daily P&L: ${daily_pl:,.2f} ({daily_pl_pct:.2f}%)\n\n")

            f.write(f"Open Positions: {len(positions)}{positions_note}\n")
            if len(positions) > 0:
                f.write("\nPosition Details:\n")
                f.write("-" * 80 + "\n")
                for pos in positions:
                    f.write(f"{pos['symbol']}: {pos['qty']} shares, "
                           f"P&L ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2f}%)\n")

        print(f"[OK] Daily report saved: {report_file.name}")


def main():
    """Main function for testing"""
    # Guard: this script flattens ALL positions before buying. Under the
    # staggered strategy that would destroy open multi-day tranches and
    # desync trading_logs/position_tranches.json. Mirror of the guard in
    # run_staggered_trading.py.
    try:
        import config_trading
        if config_trading.STRATEGY == "staggered":
            print("[ABORT] config_trading.STRATEGY = 'staggered'.")
            print("        Use run_staggered_trading.py; running the legacy trader")
            print("        would flatten all staggered tranches. Set STRATEGY='legacy'")
            print("        only if you intend to abandon the staggered book.")
            return
    except ImportError:
        pass

    trader = AlpacaAutoTrader()

    # Print status
    trader.print_portfolio_status()

    # Load and execute signals
    signals = trader.load_signals()
    if signals:
        trader.execute_trades(signals)

        # Print status after trades
        time.sleep(3)
        trader.print_portfolio_status()


if __name__ == "__main__":
    main()
