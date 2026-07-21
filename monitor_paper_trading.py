"""
PAPER TRADING 监控仪表板

实时查看Alpaca Paper Trading账户状态和历史表现
"""

import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
except ImportError:
    print("[错误] Alpaca库未安装!")
    print("运行: pip install alpaca-py")
    sys.exit(1)

from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


class PaperTradingMonitor:
    """Paper Trading监控系统"""

    def __init__(self):
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER
        )

        self.project_dir = Path(__file__).parent
        self.logs_dir = self.project_dir / "trading_logs"
        self.reports_dir = self.project_dir / "paper_trading_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def get_account_info(self):
        """获取账户信息"""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'initial_equity': float(account.last_equity),
                'daytrade_count': int(account.daytrade_count or 0),
                'status': account.status
            }
        except Exception as e:
            print(f"[错误] 获取账户信息失败: {e}")
            return None

    def get_positions(self):
        """获取当前持仓"""
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
            print(f"[错误] 获取持仓失败: {e}")
            return []

    def get_orders_history(self, limit=50):
        """获取历史订单"""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit
            )
            orders = self.trading_client.get_orders(request)

            result = []
            for order in orders:
                result.append({
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty) if order.qty else 0,
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'order_type': order.order_type,
                    'status': order.status,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                    'created_at': order.created_at,
                    'filled_at': order.filled_at
                })
            return result
        except Exception as e:
            print(f"[错误] 获取订单历史失败: {e}")
            return []

    def load_trading_logs(self):
        """加载所有交易日志"""
        if not self.logs_dir.exists():
            return []

        log_files = sorted(self.logs_dir.glob("trades_*.json"))
        logs = []

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    logs.append({
                        'file': log_file.name,
                        'timestamp': data.get('timestamp'),
                        'trades': data.get('trades_executed', []),
                        'account': data.get('account_snapshot', {})
                    })
            except Exception as e:
                print(f"[警告] 读取日志失败 {log_file.name}: {e}")

        return logs

    def calculate_performance_metrics(self, logs):
        """计算历史性能指标"""
        if len(logs) == 0:
            return {}

        # 提取每日资金变化
        equity_history = []
        for log in logs:
            if 'account' in log and 'portfolio_value' in log['account']:
                equity_history.append({
                    'date': pd.to_datetime(log['timestamp']),
                    'equity': log['account']['portfolio_value']
                })

        if len(equity_history) < 2:
            return {}

        df = pd.DataFrame(equity_history).sort_values('date')

        # 计算收益率
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity

        # 计算每日收益率
        df['daily_return'] = df['equity'].pct_change()

        # 夏普比率
        avg_daily_return = df['daily_return'].mean()
        std_daily_return = df['daily_return'].std()
        sharpe = (avg_daily_return / std_daily_return * (252 ** 0.5)) if std_daily_return > 0 else 0

        # 最大回撤
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()

        # 统计所有交易
        all_trades = []
        for log in logs:
            all_trades.extend(log['trades'])

        winning_trades = len([t for t in all_trades if t.get('amount', 0) > 0])
        total_trades = len(all_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'days_active': len(df)
        }

    def print_dashboard(self):
        """打印监控仪表板"""
        print("\n" + "="*80)
        print("📊 PAPER TRADING 监控仪表板")
        print("="*80)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模式: {'模拟交易 (Paper Trading)' if ALPACA_PAPER else '真实交易 (Live)'}")
        print()

        # 账户信息
        print("【账户状态】")
        print("-"*80)
        account = self.get_account_info()

        if account:
            daily_pl = account['equity'] - account['initial_equity']
            daily_pl_pct = (daily_pl / account['initial_equity'] * 100) if account['initial_equity'] > 0 else 0
            pl_sign = '+' if daily_pl >= 0 else ''

            print(f"总资产:        ${account['equity']:>12,.2f}")
            print(f"现金:          ${account['cash']:>12,.2f}")
            print(f"持仓市值:      ${account['equity'] - account['cash']:>12,.2f}")
            print(f"今日盈亏:      {pl_sign}${daily_pl:>11,.2f} ({pl_sign}{daily_pl_pct:.2f}%)")
            print(f"购买力:        ${account['buying_power']:>12,.2f}")
            print(f"账户状态:      {account['status']}")
        else:
            print("无法获取账户信息")

        print()

        # 持仓信息
        print("【当前持仓】")
        print("-"*80)
        positions = self.get_positions()

        if len(positions) > 0:
            print(f"{'代码':<8} {'数量':<8} {'成本':<10} {'现价':<10} {'盈亏$':<12} {'盈亏%':<8}")
            print("-"*70)

            total_pl = 0
            for pos in positions:
                pl_sign = '+' if pos['unrealized_pl'] >= 0 else ''
                print(f"{pos['symbol']:<8} {pos['qty']:<8.0f} "
                      f"${pos['avg_entry_price']:<9.2f} ${pos['current_price']:<9.2f} "
                      f"{pl_sign}${pos['unrealized_pl']:<10.2f} {pl_sign}{pos['unrealized_plpc']:.2f}%")
                total_pl += pos['unrealized_pl']

            print("-"*70)
            pl_sign = '+' if total_pl >= 0 else ''
            print(f"{'合计':<36} {pl_sign}${total_pl:<10.2f}")
        else:
            print("当前无持仓")

        print()

        # 历史表现
        print("【历史表现】")
        print("-"*80)
        logs = self.load_trading_logs()

        if len(logs) > 0:
            print(f"交易日志数: {len(logs)} 个")

            metrics = self.calculate_performance_metrics(logs)

            if metrics:
                print(f"\n自开始以来:")
                print(f"  活跃天数:      {metrics['days_active']}")
                print(f"  初始资金:      ${metrics['initial_equity']:,.2f}")
                print(f"  当前资金:      ${metrics['final_equity']:,.2f}")
                print(f"  总收益率:      {metrics['total_return_pct']:.2f}%")
                print(f"  夏普比率:      {metrics['sharpe_ratio']:.2f}")
                print(f"  最大回撤:      {metrics['max_drawdown_pct']:.2f}%")
                print(f"  总交易次数:    {metrics['total_trades']}")
                print(f"  胜率:          {metrics['win_rate']:.2f}%")
        else:
            print("暂无交易历史")

        print()

        # 最近订单
        print("【最近订单】")
        print("-"*80)
        orders = self.get_orders_history(limit=10)

        if len(orders) > 0:
            for order in orders[:5]:
                filled_time = order['filled_at'].strftime('%m-%d %H:%M') if order['filled_at'] else 'N/A'
                print(f"{order['symbol']:<6} {order['side']:<4} {order['filled_qty']:<6.0f} @ "
                      f"${order['filled_avg_price']:<7.2f} | {order['status']:<8} | {filled_time}")
        else:
            print("暂无订单历史")

        print()
        print("="*80)

    def save_daily_snapshot(self):
        """保存每日快照"""
        account = self.get_account_info()
        positions = self.get_positions()

        if not account:
            print("[错误] 无法获取账户信息")
            return

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'account': account,
            'positions': positions
        }

        # 保存到文件
        snapshot_file = self.reports_dir / f"snapshot_{datetime.now().strftime('%Y%m%d')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f"✓ 每日快照已保存: {snapshot_file.name}")

    def generate_performance_report(self):
        """生成性能报告"""
        logs = self.load_trading_logs()

        if len(logs) < 2:
            print("[提示] 交易数据不足，至少需要2天的数据")
            return

        print("\n生成性能报告...")

        # 提取资金曲线
        equity_data = []
        for log in logs:
            if 'account' in log and 'portfolio_value' in log['account']:
                equity_data.append({
                    'date': pd.to_datetime(log['timestamp']),
                    'equity': log['account']['portfolio_value']
                })

        df = pd.DataFrame(equity_data).sort_values('date')

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 资金曲线
        axes[0].plot(df['date'], df['equity'], linewidth=2, color='#2E86AB', label='Paper Trading')
        axes[0].axhline(y=df['equity'].iloc[0], color='gray', linestyle='--',
                       alpha=0.5, label='Initial Capital')
        axes[0].set_title('Paper Trading - Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 回撤
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax'] * 100

        axes[1].fill_between(df['date'], df['drawdown'], 0, color='#A23B72', alpha=0.5)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        report_file = self.reports_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(report_file, dpi=150, bbox_inches='tight')
        print(f"✓ 性能报告已保存: {report_file.name}")

        # 计算并保存指标
        metrics = self.calculate_performance_metrics(logs)
        metrics_file = self.reports_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ 性能指标已保存: {metrics_file.name}")

    def compare_with_backtest(self):
        """对比Paper Trading vs 回测结果"""
        print("\n【Paper Trading vs 回测对比】")
        print("-"*80)

        # 加载Paper Trading数据
        logs = self.load_trading_logs()
        if len(logs) < 2:
            print("Paper Trading数据不足，无法对比")
            return

        pt_metrics = self.calculate_performance_metrics(logs)

        # 加载回测数据
        backtest_dir = self.project_dir / "research" / "backtest" / "results"
        backtest_files = sorted(backtest_dir.glob("metrics_*.json"))

        if len(backtest_files) == 0:
            print("未找到回测数据")
            return

        # 加载最新的回测结果
        with open(backtest_files[-1], 'r') as f:
            bt_metrics = json.load(f)

        # 对比
        print(f"\n{'指标':<20} {'回测':<15} {'Paper Trading':<15} {'差异':<10}")
        print("-"*70)

        def compare_metric(name, bt_key, pt_key, is_pct=True):
            bt_val = bt_metrics.get(bt_key, 0)
            pt_val = pt_metrics.get(pt_key, 0)
            diff = pt_val - bt_val

            if is_pct:
                print(f"{name:<20} {bt_val:>13.2f}%  {pt_val:>13.2f}%  {diff:>8.2f}%")
            else:
                print(f"{name:<20} {bt_val:>14.2f}  {pt_val:>14.2f}  {diff:>9.2f}")

        compare_metric("收益率", "total_return_pct", "total_return_pct", True)
        compare_metric("夏普比率", "sharpe_ratio", "sharpe_ratio", False)
        compare_metric("最大回撤", "max_drawdown_pct", "max_drawdown_pct", True)
        compare_metric("胜率", "win_rate", "win_rate", True)

        print("-"*70)
        print("\n提示:")
        print("  - 如果差异在±5%以内，说明策略表现稳定")
        print("  - 如果Paper Trading明显差于回测，可能存在:")
        print("    1. 回测过拟合")
        print("    2. 市场环境变化")
        print("    3. 滑点和成本影响")


def main():
    """主函数"""
    monitor = PaperTradingMonitor()

    # 打印仪表板
    monitor.print_dashboard()

    # 保存每日快照
    monitor.save_daily_snapshot()

    # 生成性能报告
    monitor.generate_performance_report()

    # 对比回测
    monitor.compare_with_backtest()


if __name__ == "__main__":
    main()
