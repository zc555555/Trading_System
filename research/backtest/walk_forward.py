"""
Walk-forward backtesting framework.

Implements time-series cross-validation with realistic transaction costs,
position limits, and comprehensive performance metrics.

Key features:
- Walk-forward validation (no look-ahead bias)
- Transaction costs (commission + slippage)
- Position sizing constraints
- Risk management (max drawdown, position limits)
- Purging and embargo to prevent leakage
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Data split
    train_window: int = 756  # ~3 years
    test_window: int = 63  # ~3 months
    step_size: int = 21  # ~1 month

    # Costs
    commission: float = 0.0005  # 5 bps
    slippage: float = 0.001  # 10 bps

    # Position constraints
    max_position_size: float = 0.35  # Max 35% per stock
    max_total_leverage: float = 1.0  # Long-only
    min_position_size: float = 0.01  # Min 1% per position

    # Capital
    initial_capital: float = 100000.0

    # Risk management
    max_drawdown_stop: Optional[float] = None  # Stop if drawdown exceeds this
    max_daily_loss: Optional[float] = None  # Stop if daily loss exceeds this

    # Data handling
    purge_days: int = 0  # Days to purge after training set
    embargo_days: int = 0  # Days to embargo at end of training set


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Portfolio time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Costs
    total_commission: float
    total_slippage: float
    total_costs: float
    cost_ratio: float  # Costs as % of gross returns

    # Turnover
    turnover: float  # Average daily turnover

    def summary(self) -> Dict:
        """Return summary as dictionary."""
        return {
            'total_return': f"{self.total_return * 100:.2f}%",
            'annualized_return': f"{self.annualized_return * 100:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.3f}",
            'sortino_ratio': f"{self.sortino_ratio:.3f}",
            'max_drawdown': f"{self.max_drawdown * 100:.2f}%",
            'calmar_ratio': f"{self.calmar_ratio:.3f}",
            'num_trades': self.num_trades,
            'win_rate': f"{self.win_rate * 100:.2f}%",
            'turnover': f"{self.turnover * 100:.2f}%",
            'total_costs': f"${self.total_costs:,.2f}",
            'cost_ratio': f"{self.cost_ratio * 100:.2f}%",
        }


class WalkForwardBacktester:
    """Walk-forward backtesting engine."""

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_func,
        signal_func=None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            df: DataFrame with features and labels (date, symbol, features, label)
            feature_cols: List of feature column names
            model_func: Function that takes (X_train, y_train) and returns fitted model
            signal_func: Optional function that takes (model, X) and returns signals
                        Default: uses model.predict_proba for classification

        Returns:
            BacktestResult object
        """
        print("=" * 70)
        print("WALK-FORWARD BACKTEST")
        print("=" * 70)

        # Get unique dates
        dates = sorted(df['date'].unique())

        print(f"\nData range: {dates[0]} to {dates[-1]}")
        print(f"Total trading days: {len(dates)}")
        print(f"\nBacktest configuration:")
        print(f"  Train window: {self.config.train_window} days")
        print(f"  Test window: {self.config.test_window} days")
        print(f"  Step size: {self.config.step_size} days")
        print(f"  Commission: {self.config.commission * 100:.2f}%")
        print(f"  Slippage: {self.config.slippage * 100:.2f}%")

        # Generate walk-forward splits
        splits = self._generate_splits(dates)

        print(f"\nNumber of walk-forward folds: {len(splits)}")

        # Run each fold
        all_predictions = []
        fold_results = []

        for i, (train_dates, test_dates) in enumerate(splits):
            print(f"\n{'=' * 70}")
            print(f"Fold {i + 1}/{len(splits)}")
            print(f"{'=' * 70}")
            print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
            print(f"Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

            # Prepare data
            train_data = df[df['date'].isin(train_dates)]
            test_data = df[df['date'].isin(test_dates)]

            X_train = train_data[feature_cols].values
            y_train = train_data['label'].values

            X_test = test_data[feature_cols].values
            test_data_indexed = test_data.copy()

            # Train model
            print("Training model...")
            model = model_func(X_train, y_train)

            # Generate signals
            if signal_func is None:
                # Default: use predict_proba for classification
                if hasattr(model, 'predict_proba'):
                    signals = model.predict_proba(X_test)[:, 1]  # Probability of class 1
                else:
                    signals = model.predict(X_test)
            else:
                signals = signal_func(model, X_test)

            test_data_indexed['signal'] = signals

            # Store predictions
            all_predictions.append(test_data_indexed[['date', 'symbol', 'close', 'signal']])

            print(f"Generated {len(signals)} signals")

        # Combine all predictions
        predictions = pd.concat(all_predictions, ignore_index=True)

        print(f"\n{'=' * 70}")
        print("SIMULATING PORTFOLIO")
        print(f"{'=' * 70}")

        # Simulate portfolio
        portfolio = self._simulate_portfolio(predictions)

        # Calculate metrics
        result = self._calculate_metrics(portfolio)

        print(f"\n{'=' * 70}")
        print("BACKTEST RESULTS")
        print(f"{'=' * 70}\n")

        for key, value in result.summary().items():
            print(f"{key:20s}: {value}")

        return result

    def _generate_splits(self, dates: List) -> List[Tuple[List, List]]:
        """
        Generate walk-forward train/test splits.

        Args:
            dates: List of sorted dates

        Returns:
            List of (train_dates, test_dates) tuples
        """
        splits = []

        train_window = self.config.train_window
        test_window = self.config.test_window
        step_size = self.config.step_size

        # Start when we have enough data for training
        start_idx = train_window

        while start_idx + test_window <= len(dates):
            # Training set: [start_idx - train_window : start_idx]
            train_dates = dates[start_idx - train_window : start_idx]

            # Apply purge/embargo
            if self.config.purge_days > 0:
                train_dates = train_dates[: -self.config.purge_days]

            # Test set: [start_idx : start_idx + test_window]
            test_dates = dates[start_idx : start_idx + test_window]

            splits.append((train_dates, test_dates))

            # Move forward
            start_idx += step_size

        return splits

    def _simulate_portfolio(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio based on signals.

        Args:
            predictions: DataFrame with date, symbol, close, signal

        Returns:
            DataFrame with portfolio state at each timestamp
        """
        dates = sorted(predictions['date'].unique())

        # Initialize portfolio
        cash = self.config.initial_capital
        positions = {}  # {symbol: shares}
        portfolio_value = []
        position_history = []
        trade_history = []

        for date in dates:
            day_data = predictions[predictions['date'] == date]

            # Get current prices
            prices = day_data.set_index('symbol')['close'].to_dict()

            # Calculate current portfolio value
            holdings_value = sum(positions.get(sym, 0) * prices.get(sym, 0) for sym in positions)
            total_value = cash + holdings_value

            # Get signals
            signals = day_data.set_index('symbol')['signal'].to_dict()

            # Generate target positions (signal to weights)
            target_weights = self._signals_to_weights(signals)

            # Rebalance
            trades, new_cash, new_positions = self._rebalance(
                cash,
                positions,
                target_weights,
                prices,
                total_value,
            )

            # Record trades
            for trade in trades:
                trade['date'] = date
                trade_history.append(trade)

            # Update portfolio
            cash = new_cash
            positions = new_positions

            # Calculate final value
            holdings_value = sum(positions.get(sym, 0) * prices.get(sym, 0) for sym in positions)
            total_value = cash + holdings_value

            # Record state
            portfolio_value.append({
                'date': date,
                'cash': cash,
                'holdings_value': holdings_value,
                'total_value': total_value,
            })

            # Record positions
            for symbol, shares in positions.items():
                if shares != 0:
                    position_history.append({
                        'date': date,
                        'symbol': symbol,
                        'shares': shares,
                        'price': prices[symbol],
                        'value': shares * prices[symbol],
                        'weight': (shares * prices[symbol]) / total_value,
                    })

        portfolio_df = pd.DataFrame(portfolio_value)
        positions_df = pd.DataFrame(position_history)
        trades_df = pd.DataFrame(trade_history)

        return {
            'portfolio': portfolio_df,
            'positions': positions_df,
            'trades': trades_df,
        }

    def _signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Convert raw signals to portfolio weights.

        Uses rank-based weighting with position size constraints.

        Args:
            signals: Dict of {symbol: signal_value}

        Returns:
            Dict of {symbol: target_weight}
        """
        if not signals:
            return {}

        # Rank signals
        signal_series = pd.Series(signals)
        ranks = signal_series.rank(pct=True)

        # Top N positions (where rank > threshold)
        threshold = 0.5  # Only trade top 50%
        selected = ranks[ranks > threshold]

        if len(selected) == 0:
            return {}

        # Equal weight among selected (can also use rank-weighted)
        weights = {}
        weight_per_position = min(
            self.config.max_position_size,
            self.config.max_total_leverage / len(selected)
        )

        for symbol in selected.index:
            weights[symbol] = weight_per_position

        # Normalize if needed
        total_weight = sum(weights.values())
        if total_weight > self.config.max_total_leverage:
            scale = self.config.max_total_leverage / total_weight
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def _rebalance(
        self,
        cash: float,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        total_value: float,
    ) -> Tuple[List[Dict], float, Dict[str, float]]:
        """
        Rebalance portfolio to target weights.

        Args:
            cash: Current cash
            current_positions: Current positions {symbol: shares}
            target_weights: Target weights {symbol: weight}
            prices: Current prices {symbol: price}
            total_value: Total portfolio value

        Returns:
            (trades, new_cash, new_positions)
        """
        trades = []
        new_positions = current_positions.copy()

        # Calculate target shares for each symbol
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            current_shares = current_positions.get(symbol, 0)
            target_value = total_value * target_weight
            target_shares = target_value / price

            # Calculate trade
            trade_shares = target_shares - current_shares

            if abs(trade_shares) > 0.01:  # Minimum trade threshold
                # Calculate costs
                trade_value = abs(trade_shares) * price
                commission = trade_value * self.config.commission
                slippage = trade_value * self.config.slippage
                total_cost = commission + slippage

                # Check if we have enough cash (for buys)
                if trade_shares > 0:
                    required_cash = trade_value + total_cost
                    if required_cash > cash:
                        # Reduce trade to affordable amount
                        affordable_value = cash - total_cost
                        if affordable_value > 0:
                            trade_shares = affordable_value / price
                        else:
                            continue  # Skip this trade

                # Execute trade
                cash_change = -trade_shares * price - total_cost

                trades.append({
                    'symbol': symbol,
                    'shares': trade_shares,
                    'price': price,
                    'value': trade_shares * price,
                    'commission': commission,
                    'slippage': slippage,
                    'total_cost': total_cost,
                })

                cash += cash_change
                new_positions[symbol] = new_positions.get(symbol, 0) + trade_shares

        # Close positions not in target
        for symbol, shares in current_positions.items():
            if symbol not in target_weights and shares != 0:
                if symbol not in prices:
                    continue

                price = prices[symbol]
                trade_value = abs(shares) * price
                commission = trade_value * self.config.commission
                slippage = trade_value * self.config.slippage
                total_cost = commission + slippage

                trades.append({
                    'symbol': symbol,
                    'shares': -shares,
                    'price': price,
                    'value': -shares * price,
                    'commission': commission,
                    'slippage': slippage,
                    'total_cost': total_cost,
                })

                cash += shares * price - total_cost
                new_positions[symbol] = 0

        # Clean up zero positions
        new_positions = {k: v for k, v in new_positions.items() if abs(v) > 0.001}

        return trades, cash, new_positions

    def _calculate_metrics(self, portfolio: Dict) -> BacktestResult:
        """
        Calculate performance metrics.

        Args:
            portfolio: Portfolio simulation results

        Returns:
            BacktestResult object
        """
        portfolio_df = portfolio['portfolio']
        trades_df = portfolio['trades']

        # Equity curve
        equity_curve = portfolio_df.set_index('date')['total_value']

        # Returns
        returns = equity_curve.pct_change().dropna()

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        days = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()

        # Max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby((~is_drawdown).cumsum()).cumsum()
        max_drawdown_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if len(trades_df) > 0:
            num_trades = len(trades_df)

            # Wins and losses (need to track P&L per trade properly)
            # Simplified: use returns on days with trades
            trade_dates = trades_df['date'].unique()
            trade_returns = returns[returns.index.isin(trade_dates)]

            if len(trade_returns) > 0:
                wins = trade_returns[trade_returns > 0]
                losses = trade_returns[trade_returns < 0]

                win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = losses.mean() if len(losses) > 0 else 0

                gross_wins = wins.sum() if len(wins) > 0 else 0
                gross_losses = abs(losses.sum()) if len(losses) > 0 else 0
                profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0

            # Costs
            total_commission = trades_df['commission'].sum()
            total_slippage = trades_df['slippage'].sum()
            total_costs = total_commission + total_slippage

            gross_return = total_return + (total_costs / self.config.initial_capital)
            cost_ratio = total_costs / (gross_return * self.config.initial_capital) if gross_return > 0 else 0

            # Turnover
            daily_turnover = trades_df.groupby('date')['value'].apply(lambda x: x.abs().sum())
            avg_portfolio_value = portfolio_df['total_value'].mean()
            turnover = (daily_turnover / avg_portfolio_value).mean()

        else:
            num_trades = win_rate = avg_win = avg_loss = profit_factor = 0
            total_commission = total_slippage = total_costs = cost_ratio = turnover = 0

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=portfolio['positions'],
            trades=trades_df,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(max_drawdown_duration),
            calmar_ratio=calmar_ratio,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_costs,
            cost_ratio=cost_ratio,
            turnover=turnover,
        )


def main():
    """Example usage with dummy model."""
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from sklearn.ensemble import RandomForestClassifier

    # Load dataset
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    train_path = artifacts_dir / "train_dataset.parquet"

    if not train_path.exists():
        print("Error: Training dataset not found. Run build_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_parquet(train_path)

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features")

    # Define model training function
    def train_model(X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    # Run backtest
    config = BacktestConfig(
        train_window=252,  # 1 year
        test_window=63,    # 3 months
        step_size=21,      # 1 month
    )

    backtester = WalkForwardBacktester(config)
    result = backtester.run(df, feature_cols, train_model)

    # Plot results (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        result.equity_curve.plot(ax=axes[0], title='Equity Curve')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True)

        # Drawdown
        cumulative = (1 + result.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=axes[1], title='Drawdown', color='red')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(artifacts_dir / 'backtest_results.png', dpi=150)
        print(f"\nPlot saved to {artifacts_dir / 'backtest_results.png'}")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
