"""
RUN BACKTEST - Multi-Factor Strategy

Walk-forward backtesting of the multi-factor stock prediction strategy.
Tests strategy performance on historical data (2023-2025) without look-ahead bias.

Usage:
    python run_backtest_multi_factor.py
"""

import sys
import io

# Fix Windows console encoding for Chinese characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine
from factors.factor_definitions import FACTOR_GROUPS
from factors.factor_weighting import load_effective_factor_weights

# W1.5: blend factors with the same effective weights used by production signals.
FACTOR_WEIGHTS = load_effective_factor_weights()


class MultiFactorBacktest:
    """Backtest runner for multi-factor strategy"""

    def __init__(self, start_date: str = '2023-01-01',
                 end_date: str = '2025-12-31',
                 initial_capital: float = 100000,
                 top_n_stocks: int = 10):
        """
        Initialize backtest

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting capital
            top_n_stocks: Number of top stocks to trade each day
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.top_n_stocks = top_n_stocks

        self.project_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.project_dir / "research" / "data"
        self.artifacts_dir = self.project_dir / "research" / "artifacts"
        self.models_dir = self.project_dir / "research" / "models"
        self.output_dir = self.project_dir / "research" / "backtest" / "results"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print("="*80)
        print("多因子策略回测系统")
        print("="*80)
        print(f"回测时间范围: {start_date} 至 {end_date}")
        print(f"初始资金: ${initial_capital:,.0f}")
        print(f"每日交易股票数: {top_n_stocks}")
        print()

    def load_data(self):
        """Load historical data"""
        print("[1/6] 加载历史数据...")

        data_file = self.data_dir / "stocks_with_time_windows.parquet"
        if not data_file.exists():
            print(f"[错误] 数据文件不存在: {data_file}")
            return None

        df = pd.read_parquet(data_file)
        df['date'] = pd.to_datetime(df['date'])

        # Remove timezone if present for comparison
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        # Filter date range
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        print(f"  ✓ 加载完成: {len(df)} 行数据")
        print(f"  ✓ 股票数量: {df['symbol'].nunique()}")
        print(f"  ✓ 日期范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
        print()

        return df

    def load_models(self):
        """Load trained factor models"""
        import pickle

        print("[2/6] 加载多因子模型...")

        factor_ensembles = {}

        for factor_name in FACTOR_GROUPS.keys():
            # Look in artifacts directory for .pkl files
            model_file = self.artifacts_dir / f"ensemble_{factor_name}.pkl"

            if not model_file.exists():
                print(f"  [警告] 模型不存在: {factor_name}")
                continue

            with open(model_file, 'rb') as f:
                ensemble_data = pickle.load(f)

            factor_ensembles[factor_name] = ensemble_data
            print(f"  ✓ {factor_name}: 已加载")

        print(f"  ✓ 共加载 {len(factor_ensembles)} 个因子模型")
        print()

        return factor_ensembles

    def generate_signals_for_date(self, df: pd.DataFrame, date: pd.Timestamp,
                                   factor_ensembles: dict) -> list:
        """
        Generate trading signals for a specific date using only past data

        Args:
            df: Full dataframe
            date: Target date
            factor_ensembles: Loaded factor models

        Returns:
            List of signal dicts with 'symbol' and 'position_pct'
        """
        # Get data up to this date (avoid look-ahead bias)
        historical_df = df[df['date'] <= date].copy()

        if len(historical_df) == 0:
            return []

        # Get latest data for each symbol
        latest_df = historical_df.sort_values('date').groupby('symbol').tail(1)

        # Calculate predictions from each factor
        factor_predictions = {}

        for factor_name, ensemble_data in factor_ensembles.items():
            if factor_name not in FACTOR_GROUPS:
                continue

            # Get feature columns from the trained model
            feature_cols = ensemble_data.get('feature_cols', [])

            # Check which features are available
            available_features = [f for f in feature_cols if f in latest_df.columns]

            if len(available_features) == 0:
                continue

            # Prepare feature matrix
            X_df = latest_df[available_features].copy()

            # Handle missing values
            X_df = X_df.fillna(0)
            X = X_df.values

            # Get predictions from each model in the ensemble
            models = ensemble_data['models']
            weights = ensemble_data['weights']

            ensemble_predictions = np.zeros(len(X))

            for model, weight in zip(models, weights):
                try:
                    pred = model.predict(X)
                    ensemble_predictions += pred * weight
                except Exception as e:
                    # Skip this model if prediction fails
                    continue

            factor_predictions[factor_name] = pd.Series(ensemble_predictions, index=latest_df.index)

        if len(factor_predictions) == 0:
            return []

        # Combine factor predictions with weights
        combined_score = pd.Series(0.0, index=latest_df.index)

        for factor_name, predictions in factor_predictions.items():
            weight = FACTOR_WEIGHTS.get(factor_name, 0.1)
            combined_score += predictions * weight

        # Normalize scores
        combined_score = (combined_score - combined_score.mean()) / (combined_score.std() + 1e-8)

        # Get top N stocks
        top_indices = combined_score.nlargest(self.top_n_stocks).index
        top_stocks = latest_df.loc[top_indices, 'symbol'].values

        # Calculate position sizes (equal weight for simplicity)
        position_pct = 100 / self.top_n_stocks

        signals = []
        for symbol in top_stocks:
            signals.append({
                'symbol': symbol,
                'position_pct': position_pct,
                'score': combined_score[latest_df['symbol'] == symbol].values[0]
            })

        return signals

    def run_backtest(self):
        """Run the backtest simulation"""
        # Load data and models
        df = self.load_data()
        if df is None:
            return None

        factor_ensembles = self.load_models()
        if len(factor_ensembles) == 0:
            print("[错误] 没有加载到任何因子模型")
            return None

        print("[3/6] 开始回测模拟...")

        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            max_position_pct=0.15,
            stop_loss_pct=0.025,
            holding_days=1
        )

        # Get all unique trading dates
        trading_dates = sorted(df['date'].unique())
        total_dates = len(trading_dates)

        print(f"  ✓ 交易日总数: {total_dates}")
        print()

        # Walk forward through each date
        for i, date in enumerate(trading_dates):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  处理日期: {date.date()} [{i+1}/{total_dates}]")

            # Generate signals using only past data
            signals = self.generate_signals_for_date(df, date, factor_ensembles)

            # Get current prices for this date
            date_df = df[df['date'] == date]
            current_prices = dict(zip(date_df['symbol'], date_df['close']))

            # Execute signals
            if len(signals) > 0:
                engine.execute_signals(date.strftime('%Y-%m-%d'), signals, current_prices)

            # Update portfolio at end of day
            engine.update_daily(date.strftime('%Y-%m-%d'), current_prices)

        print(f"  ✓ 回测完成!")
        print()

        return engine

    def calculate_performance(self, engine: BacktestEngine):
        """Calculate and display performance metrics"""
        print("[4/6] 计算性能指标...")

        metrics = engine.calculate_metrics()

        if len(metrics) == 0:
            print("  [错误] 无法计算性能指标")
            return metrics

        print()
        print("="*80)
        print("回测结果")
        print("="*80)
        print(f"初始资金:      ${metrics['initial_capital']:>12,.2f}")
        print(f"最终资金:      ${metrics['final_equity']:>12,.2f}")
        print(f"总收益率:      {metrics['total_return_pct']:>12.2f}%")
        print(f"年化收益率:    {metrics['annualized_return_pct']:>12.2f}%")
        print(f"夏普比率:      {metrics['sharpe_ratio']:>12.2f}")
        print(f"最大回撤:      {metrics['max_drawdown_pct']:>12.2f}%")
        print()
        print(f"总交易次数:    {metrics['total_trades']:>12}")
        print(f"盈利次数:      {metrics['winning_trades']:>12}")
        print(f"亏损次数:      {metrics['losing_trades']:>12}")
        print(f"胜率:          {metrics['win_rate']:>12.2f}%")
        print(f"平均盈利:      ${metrics['avg_win']:>11,.2f}")
        print(f"平均亏损:      ${metrics['avg_loss']:>11,.2f}")
        print(f"盈亏比:        {metrics['profit_factor']:>12.2f}")
        print("="*80)
        print()

        return metrics

    def create_visualizations(self, engine: BacktestEngine):
        """Create visualization charts"""
        print("[5/6] 生成可视化图表...")

        equity_df = engine.get_equity_curve_df()

        if len(equity_df) == 0:
            print("  [警告] 没有数据可供可视化")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Equity curve
        axes[0].plot(pd.to_datetime(equity_df['date']), equity_df['equity'],
                    linewidth=2, color='#2E86AB', label='策略资金曲线')
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--',
                       alpha=0.5, label='初始资金')
        axes[0].set_title('资金曲线 (Equity Curve)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('资金 ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100

        axes[1].fill_between(pd.to_datetime(equity_df['date']), drawdown, 0,
                            color='#A23B72', alpha=0.5)
        axes[1].set_title('回撤曲线 (Drawdown)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('回撤 (%)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ 图表已保存: {output_file.name}")
        print()

    def save_results(self, engine: BacktestEngine, metrics: dict):
        """Save detailed results to files"""
        print("[6/6] 保存回测结果...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ 性能指标: {metrics_file.name}")

        # Save equity curve
        equity_df = engine.get_equity_curve_df()
        equity_file = self.output_dir / f"equity_curve_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        print(f"  ✓ 资金曲线: {equity_file.name}")

        # Save trades
        trades_df = engine.get_trades_df()
        if len(trades_df) > 0:
            trades_file = self.output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"  ✓ 交易记录: {trades_file.name}")

        print()
        print("="*80)
        print("回测完成! 所有结果已保存到:")
        print(f"  {self.output_dir}")
        print("="*80)


def main():
    """Main function"""
    # Configure backtest parameters
    backtest = MultiFactorBacktest(
        start_date='2023-01-01',
        end_date='2025-12-31',
        initial_capital=100000,
        top_n_stocks=10
    )

    # Run backtest
    engine = backtest.run_backtest()

    if engine is None:
        print("[错误] 回测失败")
        return

    # Calculate performance
    metrics = backtest.calculate_performance(engine)

    # Create visualizations
    backtest.create_visualizations(engine)

    # Save results
    backtest.save_results(engine, metrics)


if __name__ == "__main__":
    main()
