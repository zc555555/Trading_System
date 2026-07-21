"""完整交易记录系统 - 记录所有信息用于模型精调"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf
from alpaca.trading.client import TradingClient
from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY

project_dir = Path(__file__).parent
logs_dir = project_dir / "trading_logs"
logs_dir.mkdir(exist_ok=True)

print("="*80)
print(f"完整交易记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# 初始化Alpaca客户端
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# 读取今天的交易信号
today = datetime.now().strftime('%Y%m%d')
signal_file = project_dir / "research" / "artifacts" / f"signals_multi_factor_{today}.json"

if not signal_file.exists():
    print("[INFO] 今天没有交易信号，跳过记录")
    exit(0)

with open(signal_file, 'r') as f:
    signals = json.load(f)

print(f"[OK] 加载信号: {signals['n_stocks']} 只股票")
print()

# ============================================================================
# PART 1: 记录交易信号和预测
# ============================================================================

signal_records = []

for stock in signals['stocks']:
    symbol = stock['symbol']

    record = {
        # 基本信息
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'rank': stock['rank'],

        # 预测值
        'prediction': stock['prediction'],
        'prediction_pct': stock['prediction'] * 100,
        'confidence': stock['confidence'],
        'confidence_pct': stock['confidence'] * 100,
        'position_pct': stock['position_pct'],

        # 因子贡献（原始值）
        'factor_momentum': stock['factor_contributions']['momentum'],
        'factor_trend': stock['factor_contributions']['trend'],
        'factor_volatility': stock['factor_contributions']['volatility'],
        'factor_volume': stock['factor_contributions']['volume'],
        'factor_market': stock['factor_contributions']['market'],
        'factor_alpha': stock['factor_contributions']['alpha'],

        # 因子贡献（百分比）
        'factor_momentum_pct': stock['factor_contributions']['momentum'] * 100,
        'factor_trend_pct': stock['factor_contributions']['trend'] * 100,
        'factor_volatility_pct': stock['factor_contributions']['volatility'] * 100,
        'factor_volume_pct': stock['factor_contributions']['volume'] * 100,
        'factor_market_pct': stock['factor_contributions']['market'] * 100,
        'factor_alpha_pct': stock['factor_contributions']['alpha'] * 100,
    }

    signal_records.append(record)

signals_df = pd.DataFrame(signal_records)

# 添加全局配置信息
config_record = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'timestamp': datetime.now().isoformat(),

    # 因子权重
    'weight_momentum': signals['factor_weights']['momentum'],
    'weight_trend': signals['factor_weights']['trend'],
    'weight_volatility': signals['factor_weights']['volatility'],
    'weight_volume': signals['factor_weights']['volume'],
    'weight_market': signals['factor_weights']['market'],
    'weight_alpha': signals['factor_weights']['alpha'],

    # 交易配置
    'n_top': signals['config']['n_top'],
    'min_confidence': signals['config']['min_confidence'],
    'min_stocks': signals['config']['min_stocks'],

    # 交易决策
    'should_trade': signals['should_trade'],
    'n_stocks': signals['n_stocks'],
    'method': signals['method'],
    'data_date': signals['data_date'],
}

# 保存信号记录
signal_log_file = logs_dir / "daily_signals_detailed.csv"
if signal_log_file.exists():
    try:
        existing = pd.read_csv(signal_log_file)
        combined = pd.concat([existing, signals_df], ignore_index=True)
        combined.to_csv(signal_log_file, index=False)
    except pd.errors.EmptyDataError:
        # 文件为空，直接写入
        signals_df.to_csv(signal_log_file, index=False)
else:
    signals_df.to_csv(signal_log_file, index=False)

print(f"[OK] 信号记录已保存: {len(signals_df)} 条")

# 保存配置记录
config_log_file = logs_dir / "daily_config.csv"
config_df = pd.DataFrame([config_record])
if config_log_file.exists():
    try:
        existing = pd.read_csv(config_log_file)
        combined = pd.concat([existing, config_df], ignore_index=True)
        combined.to_csv(config_log_file, index=False)
    except pd.errors.EmptyDataError:
        # 文件为空，直接写入
        config_df.to_csv(config_log_file, index=False)
else:
    config_df.to_csv(config_log_file, index=False)

print(f"[OK] 配置记录已保存")
print()

# ============================================================================
# PART 2: 记录实际市场表现（第二天运行时）
# ============================================================================

# 检查是否有昨天的信号需要更新实际表现
yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
yesterday_signals = signals_df[signals_df['date'] == yesterday] if 'date' in signals_df.columns else pd.DataFrame()

# 如果是第二天，更新昨天的实际表现
performance_records = []

print("检查是否需要更新昨天的实际表现...")

# 读取历史信号
if signal_log_file.exists():
    try:
        all_signals = pd.read_csv(signal_log_file)
    except pd.errors.EmptyDataError:
        # 文件为空，跳过更新
        print("[INFO] 信号日志为空，跳过实际表现更新")
        all_signals = None

    if all_signals is not None:
        # 找出还没有记录实际表现的信号（actual_return 为空）
        if 'actual_return' not in all_signals.columns:
            all_signals['actual_return'] = np.nan

        pending_signals = all_signals[all_signals['actual_return'].isna()].copy()
    else:
        pending_signals = pd.DataFrame()

    if len(pending_signals) > 0:
        print(f"发现 {len(pending_signals)} 条待更新记录")
        print()

        for idx, row in pending_signals.iterrows():
            symbol = row['symbol']
            signal_date = pd.to_datetime(row['date'])

            try:
                # 获取信号日期之后的市场数据
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=signal_date, period='5d')

                if len(hist) < 2:
                    print(f"  [SKIP] {symbol}: 数据不足")
                    continue

                # 信号日期的收盘价（下单时的价格）
                signal_close = float(hist.iloc[0]['Close'])

                # 下一个交易日的数据
                next_open = float(hist.iloc[1]['Open'])
                next_high = float(hist.iloc[1]['High'])
                next_low = float(hist.iloc[1]['Low'])
                next_close = float(hist.iloc[1]['Close'])
                next_volume = int(hist.iloc[1]['Volume'])

                # 价格异常检测：yfinance 偶尔返回 0 或 NaN，避免除零崩溃
                if not (signal_close > 0 and next_open > 0) or np.isnan(signal_close) or np.isnan(next_open):
                    print(f"  [SKIP] {symbol}: 价格数据异常 (signal_close={signal_close}, next_open={next_open})")
                    continue

                # 计算各种收益
                gap_pct = (next_open - signal_close) / signal_close * 100
                actual_return = (next_close - next_open) / next_open * 100
                expected_return = (next_close - signal_close) / signal_close * 100
                gap_impact = expected_return - actual_return

                # 更新记录
                all_signals.at[idx, 'signal_close'] = signal_close
                all_signals.at[idx, 'next_open'] = next_open
                all_signals.at[idx, 'next_high'] = next_high
                all_signals.at[idx, 'next_low'] = next_low
                all_signals.at[idx, 'next_close'] = next_close
                all_signals.at[idx, 'next_volume'] = next_volume

                all_signals.at[idx, 'gap_pct'] = gap_pct
                all_signals.at[idx, 'actual_return'] = actual_return
                all_signals.at[idx, 'expected_return'] = expected_return
                all_signals.at[idx, 'gap_impact'] = gap_impact

                # 预测准确性
                prediction_pct = row['prediction_pct']
                prediction_error = expected_return - prediction_pct
                direction_correct = (prediction_pct > 0 and expected_return > 0) or (prediction_pct < 0 and expected_return < 0)
                trade_profitable = actual_return > 0

                all_signals.at[idx, 'prediction_error'] = prediction_error
                all_signals.at[idx, 'prediction_direction_correct'] = direction_correct
                all_signals.at[idx, 'trade_profitable'] = trade_profitable

                # 计算实际收益金额（假设$100,000账户）
                position_size = 100000 * row['position_pct'] / 100
                shares = int(position_size / next_open)
                actual_pnl = shares * (next_close - next_open)

                all_signals.at[idx, 'position_size'] = position_size
                all_signals.at[idx, 'shares'] = shares
                all_signals.at[idx, 'actual_pnl'] = actual_pnl

                print(f"  ✓ {symbol}: 预测 {prediction_pct:+.2f}%, 实际 {actual_return:+.2f}%, P&L ${actual_pnl:+.2f}")

            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")

        # 保存更新后的数据
        all_signals.to_csv(signal_log_file, index=False)
        print()
        print(f"[OK] 已更新 {len(pending_signals)} 条实际表现记录")
else:
    print("[INFO] 没有需要更新的历史记录")

print()

# ============================================================================
# PART 3: 记录账户状态
# ============================================================================

print("记录账户状态...")

try:
    account = client.get_account()
    positions = client.get_all_positions()

    account_record = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),

        # 账户信息
        'equity': float(account.equity),
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'buying_power': float(account.buying_power),
        'initial_equity': float(account.last_equity),

        # P&L
        'daily_pnl': float(account.equity) - float(account.last_equity),
        'daily_pnl_pct': (float(account.equity) - float(account.last_equity)) / float(account.last_equity) * 100 if float(account.last_equity) > 0 else 0,

        # 持仓数量
        'n_positions': len(positions),
        'day_trade_count': int(account.daytrade_count or 0),
    }

    # 保存账户记录
    account_log_file = logs_dir / "daily_account.csv"
    account_df = pd.DataFrame([account_record])
    if account_log_file.exists():
        existing = pd.read_csv(account_log_file)
        combined = pd.concat([existing, account_df], ignore_index=True)
        combined.to_csv(account_log_file, index=False)
    else:
        account_df.to_csv(account_log_file, index=False)

    print(f"[OK] 账户记录已保存")
    print(f"  资产: ${float(account.equity):,.2f}")
    print(f"  日内P&L: ${account_record['daily_pnl']:+,.2f} ({account_record['daily_pnl_pct']:+.2f}%)")
    print(f"  持仓数: {len(positions)}")

except Exception as e:
    print(f"[WARNING] 无法记录账户状态: {e}")

print()

# ============================================================================
# PART 4: 记录持仓信息
# ============================================================================

print("记录持仓信息...")

try:
    positions = client.get_all_positions()

    if len(positions) > 0:
        position_records = []

        for pos in positions:
            position_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc) * 100,
            }
            position_records.append(position_record)

        # 保存持仓记录
        position_log_file = logs_dir / "daily_positions.csv"
        position_df = pd.DataFrame(position_records)
        if position_log_file.exists():
            existing = pd.read_csv(position_log_file)
            combined = pd.concat([existing, position_df], ignore_index=True)
            combined.to_csv(position_log_file, index=False)
        else:
            position_df.to_csv(position_log_file, index=False)

        print(f"[OK] 持仓记录已保存: {len(positions)} 个持仓")

        for pos in position_records:
            print(f"  {pos['symbol']:6} {pos['qty']:>6.0f}股 @ ${pos['avg_entry_price']:>7.2f}, "
                  f"P&L: ${pos['unrealized_pl']:>8.2f} ({pos['unrealized_plpc']:>+6.2f}%)")
    else:
        print("[INFO] 当前无持仓")

except Exception as e:
    print(f"[WARNING] 无法记录持仓信息: {e}")

print()

# ============================================================================
# 汇总信息
# ============================================================================

print("="*80)
print("【记录汇总】")
print("="*80)
print()

print("已保存以下数据文件:")
print(f"  1. 交易信号详情:      {signal_log_file}")
print(f"  2. 每日配置参数:      {config_log_file}")
print(f"  3. 账户状态记录:      {account_log_file if 'account_log_file' in locals() else 'N/A'}")
print(f"  4. 持仓信息记录:      {position_log_file if 'position_log_file' in locals() else 'N/A'}")
print()

# 统计累积记录
if signal_log_file.exists():
    try:
        all_signals = pd.read_csv(signal_log_file)
    except pd.errors.EmptyDataError:
        print("[INFO] 信号日志为空，无法生成统计")
        all_signals = None

    if all_signals is not None and len(all_signals) > 0:
        total_days = all_signals['date'].nunique()
        total_trades = len(all_signals)

        if 'actual_return' in all_signals.columns:
            completed_trades = all_signals['actual_return'].notna().sum()
            pending_trades = all_signals['actual_return'].isna().sum()

            print(f"累积统计:")
            print(f"  交易日数:     {total_days}")
            print(f"  总交易数:     {total_trades}")
            print(f"  已完成:       {completed_trades}")
            print(f"  待更新:       {pending_trades}")

            if completed_trades > 0:
                completed_df = all_signals[all_signals['actual_return'].notna()]
                avg_actual_return = completed_df['actual_return'].mean()
                win_rate = (completed_df['actual_return'] > 0).sum() / completed_trades * 100

                print()
                print(f"  平均实际收益: {avg_actual_return:+.3f}%")
                print(f"  胜率:         {win_rate:.1f}%")
    else:
        print(f"累积统计:")
        print(f"  交易日数:     {total_days}")
        print(f"  总交易数:     {total_trades}")

print()
print("="*80)
print()
print("提示:")
print("  - 所有数据将持续累积")
print("  - 一个月后运行 python analyze_full_performance.py 进行深度分析")
print("  - 或者直接将 trading_logs/ 文件夹发给我进行模型精调")
print()
print("="*80)
