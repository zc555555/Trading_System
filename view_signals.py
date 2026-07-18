"""查看今日交易信号 - 包含预期涨幅、胜率等详细信息"""
import json
from pathlib import Path
from datetime import datetime

project_dir = Path(__file__).parent
artifacts_dir = project_dir / "research" / "artifacts"

# 找到最新的信号文件
today = datetime.now().strftime('%Y%m%d')
signal_file = artifacts_dir / f"signals_multi_factor_{today}.json"

if not signal_file.exists():
    print(f"[ERROR] 今天的信号文件不存在: {signal_file}")
    print("请先运行 auto_trading.bat 生成交易信号")
    exit(1)

# 读取信号文件
with open(signal_file, 'r') as f:
    signals = json.load(f)

print("="*100)
print(f"交易信号详情 - {signals['generated_at'][:10]}")
print("="*100)
print()

# 显示配置信息
print("【模型配置】")
print(f"  方法: {signals['method']}")
print(f"  数据日期: {signals['data_date']}")
print(f"  是否交易: {'✅ 是' if signals['should_trade'] else '❌ 否'}")
print(f"  推荐股票数: {signals['n_stocks']}")
print()

# 显示因子权重
print("【因子权重】")
weights = signals['factor_weights']
for factor, weight in weights.items():
    print(f"  {factor:12} : {weight*100:5.1f}%")
print()

# 显示信号阈值
print("【交易门槛】")
config = signals['config']
print(f"  候选股票数: {config['n_top']}")
print(f"  最低置信度: {config['min_confidence']*100:.1f}%")
print(f"  最少股票数: {config['min_stocks']}")
print()

if not signals['should_trade']:
    print(f"\n⚠️  今日不交易（股票数 {signals['n_stocks']} < 最少要求 {config['min_stocks']}）")
    exit(0)

# 显示推荐股票详情
print("="*100)
print("【推荐股票详情】")
print("="*100)
print()

for i, stock in enumerate(signals['stocks'], 1):
    print(f"{'='*100}")
    print(f"排名 #{i}: {stock['symbol']}")
    print(f"{'='*100}")
    print()

    # 核心指标
    prediction_pct = stock['prediction'] * 100
    confidence_pct = stock['confidence'] * 100
    position_pct = stock['position_pct']

    print(f"  📈 预期涨幅:     {prediction_pct:>6.2f}%")
    print(f"  💪 置信度:       {confidence_pct:>6.2f}%")
    print(f"  💰 建议仓位:     {position_pct:>6.2f}%")
    print()

    # 因子贡献度
    print(f"  【各因子贡献度】")
    contributions = stock['factor_contributions']

    # 按贡献度排序
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    for factor, contrib in sorted_contributions:
        contrib_pct = contrib * 100
        bar_length = int(abs(contrib_pct) * 2)  # 每1%用2个字符
        bar = '█' * min(bar_length, 40)  # 最多40个字符

        if contrib >= 0:
            print(f"    {factor:12} : +{contrib_pct:5.2f}%  {bar}")
        else:
            print(f"    {factor:12} : {contrib_pct:6.2f}%  {bar}")

    print()

# 总结
print("="*100)
print("【投资建议总结】")
print("="*100)
print()

total_position = sum(stock['position_pct'] for stock in signals['stocks'])
avg_prediction = sum(stock['prediction'] for stock in signals['stocks']) / len(signals['stocks']) * 100

print(f"  总仓位: {total_position:.2f}%")
print(f"  平均预期涨幅: {avg_prediction:.2f}%")
print(f"  持有时间: 1天（明天21:00自动平仓）")
print()

# 风险提示
print("【风险提示】")
print(f"  ⚠️  这是Paper Trading（模拟交易），使用虚拟资金")
print(f"  ⚠️  预期涨幅基于历史数据和机器学习模型，不保证未来收益")
print(f"  ⚠️  实际收益可能与预期存在偏差")
print()

print("="*100)
print()

# 显示相关文件路径
print("【相关文件】")
print(f"  信号文件: {signal_file}")
print(f"  查看实时持仓: python check_positions.py")
print(f"  查看交易日志: trading_logs/trades_*.json")
print()
print("="*100)
