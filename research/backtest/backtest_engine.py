"""
BACKTESTING ENGINE - Multi-Factor Strategy

Walk-forward backtesting system that simulates trading strategy performance
using historical data without look-ahead bias.

Features:
- Daily walk-forward simulation
- Position tracking and P&L calculation
- Risk management (stop-loss, position limits)
- Performance metrics (Sharpe, max drawdown, win rate)
- Trade logging
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import json


class BacktestEngine:
    """Core backtesting engine for simulating strategy performance"""

    def __init__(self, initial_capital: float = 100000,
                 max_position_pct: float = 0.15,
                 stop_loss_pct: float = 0.025,
                 max_daily_loss_pct: float = 0.03,
                 holding_days: int = 1):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital in dollars
            max_position_pct: Maximum position size per stock (0.15 = 15%)
            stop_loss_pct: Stop loss percentage (0.025 = 2.5%)
            max_daily_loss_pct: Maximum daily loss before stopping (0.03 = 3%)
            holding_days: Number of days to hold each position
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.holding_days = holding_days

        # Portfolio state
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions = {}  # {symbol: {'qty': int, 'entry_price': float, 'entry_date': str}}

        # History tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def reset(self):
        """Reset portfolio state for new backtest"""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value

        Args:
            current_prices: {symbol: current_price}

        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = 0
        for symbol, pos_data in self.positions.items():
            if symbol in current_prices:
                positions_value += pos_data['qty'] * current_prices[symbol]

        return self.cash + positions_value

    def check_stop_loss(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any positions hit stop loss

        Args:
            current_prices: {symbol: current_price}

        Returns:
            List of symbols to close
        """
        symbols_to_close = []

        for symbol, pos_data in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = pos_data['entry_price']

            # Calculate loss percentage
            loss_pct = (current_price - entry_price) / entry_price

            if loss_pct <= -self.stop_loss_pct:
                symbols_to_close.append(symbol)

        return symbols_to_close

    def close_position(self, symbol: str, close_price: float, close_date: str, reason: str = 'holding_period'):
        """
        Close a position and record the trade

        Args:
            symbol: Stock symbol
            close_price: Exit price
            close_date: Exit date
            reason: 'holding_period' or 'stop_loss'
        """
        if symbol not in self.positions:
            return

        pos_data = self.positions[symbol]
        qty = pos_data['qty']
        entry_price = pos_data['entry_price']
        entry_date = pos_data['entry_date']

        # Calculate P&L
        exit_value = qty * close_price
        entry_value = qty * entry_price
        pnl = exit_value - entry_value
        pnl_pct = (close_price - entry_price) / entry_price

        # Add cash from sale
        self.cash += exit_value

        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': close_date,
            'entry_price': entry_price,
            'exit_price': close_price,
            'qty': qty,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        self.trades.append(trade)

        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Remove position
        del self.positions[symbol]

    def open_position(self, symbol: str, entry_price: float, entry_date: str,
                     position_pct: float):
        """
        Open a new position

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            entry_date: Entry date
            position_pct: Position size as percentage of portfolio (0.15 = 15%)
        """
        # Calculate quantity based on available cash
        position_size = min(self.cash * position_pct, self.cash * self.max_position_pct)
        qty = int(position_size / entry_price)

        if qty < 1:
            return  # Not enough capital

        # Calculate actual cost
        cost = qty * entry_price

        if cost > self.cash:
            return  # Not enough cash

        # Deduct cash
        self.cash -= cost

        # Add position
        self.positions[symbol] = {
            'qty': qty,
            'entry_price': entry_price,
            'entry_date': entry_date
        }

    def update_daily(self, date: str, current_prices: Dict[str, float]):
        """
        Update portfolio at end of day

        Args:
            date: Current date
            current_prices: {symbol: close_price}
        """
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value(current_prices)

        # Calculate daily return
        if len(self.equity_curve) > 0:
            prev_equity = self.equity_curve[-1]['equity']
            daily_return = (portfolio_value - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
        else:
            daily_return = 0

        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'daily_return': daily_return,
            'n_positions': len(self.positions)
        })

        self.equity = portfolio_value

        # Check for stop losses (intraday)
        stop_loss_symbols = self.check_stop_loss(current_prices)
        for symbol in stop_loss_symbols:
            if symbol in current_prices:
                self.close_position(symbol, current_prices[symbol], date, 'stop_loss')

    def close_all_positions(self, current_prices: Dict[str, float], date: str, reason: str = 'holding_period'):
        """
        Close all open positions

        Args:
            current_prices: {symbol: price}
            date: Current date
            reason: Reason for closing
        """
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in current_prices:
                self.close_position(symbol, current_prices[symbol], date, reason)

    def execute_signals(self, date: str, signals: List[Dict], current_prices: Dict[str, float]):
        """
        Execute trading signals

        Args:
            date: Trading date
            signals: List of signal dicts with 'symbol' and 'position_pct'
            current_prices: {symbol: price}
        """
        # Close all existing positions first (1-day holding strategy)
        self.close_all_positions(current_prices, date, 'holding_period')

        # Open new positions based on signals
        for signal in signals:
            symbol = signal['symbol']
            position_pct = signal.get('position_pct', 0.1) / 100  # Convert from percentage

            if symbol not in current_prices:
                continue  # Skip if no price data

            entry_price = current_prices[symbol]

            # Open position
            self.open_position(symbol, entry_price, date, position_pct)

    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics

        Returns:
            Dictionary of performance metrics
        """
        if len(self.equity_curve) == 0:
            return {}

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)

        # Total return
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Annualized return (assuming 252 trading days per year)
        n_days = len(equity_df)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns statistics
        daily_returns = np.array(self.daily_returns)
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)

        # Sharpe ratio (assuming 0% risk-free rate, annualized)
        sharpe_ratio = (avg_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0

        # Maximum drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Average trade P&L
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if self.winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if self.losing_trades > 0 else 0
            profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 and avg_loss != 0 else 0
        else:
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'n_days': n_days
        }

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        return pd.DataFrame(self.trades)
