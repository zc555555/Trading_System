"""分析亏损原因"""
import sys
import io
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_trading_history():
    print("\n" + "="*80)
    print("亏损原因分析")
    print("="*80)

    # 查找交易日志
    logs_dir = Path("trading_logs")

    if not logs_dir.exists():
        print("\n[错误] 找不到交易日志目录")
        return

    # 加载所有交易事件
    event_files = sorted(logs_dir.glob("dynamic_trading_events_*.json"))

    if not event_files:
        print("\n[INFO] 没有找到交易历史记录")
        return

    print(f"\n找到 {len(event_files)} 个交易日志文件")

    all_trades = []

    for file in event_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if 'events' in data:
                    all_trades.extend(data['events'])
        except Exception as e:
            print(f"[WARNING] 无法读取 {file.name}: {e}")

    if not all_trades:
        print("\n[INFO] 没有交易记录")
        return

    print(f"\n总共 {len(all_trades)} 条交易事件")

    # 转换为DataFrame
    df = pd.DataFrame(all_trades)

    # 统计
    print("\n" + "="*80)
    print("交易统计")
    print("="*80)

    # 按事件类型分组
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts()
        print("\n事件类型分布:")
        for event, count in event_counts.items():
            print(f"  {event}: {count}")

    # 分析盈亏
    if 'pl_amount' in df.columns and 'pl_percent' in df.columns:
        closed_trades = df[df['pl_amount'].notna()].copy()

        if len(closed_trades) > 0:
            print("\n" + "="*80)
            print("盈亏分析")
            print("="*80)

            total_pl = closed_trades['pl_amount'].sum()
            wins = closed_trades[closed_trades['pl_amount'] > 0]
            losses = closed_trades[closed_trades['pl_amount'] < 0]

            print(f"\n总盈亏: ${total_pl:+,.2f}")
            print(f"总交易次数: {len(closed_trades)}")
            print(f"盈利交易: {len(wins)} ({len(wins)/len(closed_trades)*100:.1f}%)")
            print(f"亏损交易: {len(losses)} ({len(losses)/len(closed_trades)*100:.1f}%)")

            if len(wins) > 0:
                print(f"\n平均盈利: ${wins['pl_amount'].mean():+.2f}")
                print(f"最大盈利: ${wins['pl_amount'].max():+.2f}")

            if len(losses) > 0:
                print(f"\n平均亏损: ${losses['pl_amount'].mean():+.2f}")
                print(f"最大亏损: ${losses['pl_amount'].min():+.2f}")

            # 盈利因子
            if len(losses) > 0:
                profit_factor = abs(wins['pl_amount'].sum() / losses['pl_amount'].sum())
                print(f"\n盈利因子: {profit_factor:.2f}")

                if profit_factor < 1.0:
                    print("  ⚠️ 盈利因子 < 1.0，说明亏损大于盈利")
                elif profit_factor < 1.5:
                    print("  ⚠️ 盈利因子较低，需要改进")
                else:
                    print("  ✅ 盈利因子良好")

            # 按股票分析
            print("\n" + "="*80)
            print("股票表现（前10名/后10名）")
            print("="*80)

            if 'symbol' in closed_trades.columns:
                stock_pl = closed_trades.groupby('symbol')['pl_amount'].agg(['sum', 'count', 'mean'])
                stock_pl = stock_pl.sort_values('sum', ascending=False)

                print("\n🏆 表现最好的股票:")
                for i, (symbol, row) in enumerate(stock_pl.head(10).iterrows(), 1):
                    print(f"  {i}. {symbol}: ${row['sum']:+.2f} ({int(row['count'])}笔, 平均${row['mean']:+.2f})")

                print("\n📉 表现最差的股票:")
                for i, (symbol, row) in enumerate(stock_pl.tail(10).iterrows(), 1):
                    print(f"  {i}. {symbol}: ${row['sum']:+.2f} ({int(row['count'])}笔, 平均${row['mean']:+.2f})")

            # 按退出类型分析
            if 'exit_reason' in closed_trades.columns:
                print("\n" + "="*80)
                print("退出原因分析")
                print("="*80)

                exit_pl = closed_trades.groupby('exit_reason')['pl_amount'].agg(['sum', 'count', 'mean'])
                exit_pl = exit_pl.sort_values('sum', ascending=False)

                print("\n各退出类型的盈亏:")
                for reason, row in exit_pl.iterrows():
                    avg_pct = closed_trades[closed_trades['exit_reason']==reason]['pl_percent'].mean()
                    print(f"  {reason}:")
                    print(f"    总计: ${row['sum']:+.2f}")
                    print(f"    次数: {int(row['count'])}")
                    print(f"    平均: ${row['mean']:+.2f} ({avg_pct:+.2f}%)")

            # 时间分析
            if 'timestamp' in closed_trades.columns:
                print("\n" + "="*80)
                print("时间分析")
                print("="*80)

                closed_trades['date'] = pd.to_datetime(closed_trades['timestamp']).dt.date
                daily_pl = closed_trades.groupby('date')['pl_amount'].sum().sort_values()

                print("\n最差的5天:")
                for date, pl in daily_pl.head(5).items():
                    print(f"  {date}: ${pl:+.2f}")

                print("\n最好的5天:")
                for date, pl in daily_pl.tail(5).items():
                    print(f"  {date}: ${pl:+.2f}")

    print("\n" + "="*80)
    print("主要问题诊断")
    print("="*80)

    problems = []

    if 'pl_amount' in df.columns and 'pl_percent' in df.columns:
        closed = df[df['pl_amount'].notna()]

        if len(closed) > 0:
            win_rate = len(closed[closed['pl_amount'] > 0]) / len(closed)

            if win_rate < 0.5:
                problems.append(f"❌ 胜率过低: {win_rate*100:.1f}%（应该 > 50%）")

            if len(losses) > 0 and len(wins) > 0:
                avg_win = wins['pl_amount'].mean()
                avg_loss = abs(losses['pl_amount'].mean())

                if avg_loss > avg_win:
                    problems.append(f"❌ 平均亏损(${avg_loss:.2f})大于平均盈利(${avg_win:.2f})")

                profit_factor = abs(wins['pl_amount'].sum() / losses['pl_amount'].sum())
                if profit_factor < 1.0:
                    problems.append(f"❌ 盈利因子过低: {profit_factor:.2f}（应该 > 1.5）")

    if problems:
        print("\n发现的问题:")
        for problem in problems:
            print(f"  {problem}")

        print("\n💡 改进建议:")
        print("  1. 降低交易频率，只交易高信心信号（提高min_confidence）")
        print("  2. 调整止损止盈比例（当前2.5%:2.5%，建议2%:3%）")
        print("  3. 运行归因分析找出哪些因子/股票在亏钱")
        print("  4. 重新训练模型使用最新数据")
        print("  5. 考虑加入更多过滤条件（市场状态、新闻情绪等）")
    else:
        print("\n✅ 没有发现明显问题")

    print("\n" + "="*80)
    print("\n运行归因分析获取更详细信息:")
    print("  python run_menu.py")
    print("  选择 [5] 归因分析")
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_trading_history()
