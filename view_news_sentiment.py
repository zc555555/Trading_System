"""查看今日交易股票的新闻情绪"""
import json
from pathlib import Path
from datetime import datetime
from news_sentiment_analyzer import NewsSentimentAnalyzer

project_dir = Path(__file__).parent

# 读取今日交易信号
today = datetime.now().strftime('%Y%m%d')
signal_file = project_dir / "research" / "artifacts" / f"signals_multi_factor_{today}.json"

if not signal_file.exists():
    print(f"[INFO] 今天没有交易信号文件")
    print("使用默认股票测试...")
    symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
else:
    with open(signal_file, 'r') as f:
        signals = json.load(f)

    symbols = [stock['symbol'] for stock in signals['stocks']]
    print(f"[OK] 找到今日交易信号: {len(symbols)} 只股票")

print()
print("="*80)
print(f"实时新闻情绪分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# 初始化分析器
analyzer = NewsSentimentAnalyzer()

# 获取情绪
print(f"正在获取 {len(symbols)} 只股票的实时新闻...")
sentiments = analyzer.get_batch_sentiment(symbols, delay=1.0)

# 显示摘要
analyzer.display_summary(sentiments)

# 保存结果
output_file = project_dir / "trading_logs" / f"news_sentiment_{today}.json"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(sentiments, f, indent=2)

print(f"[OK] 结果已保存: {output_file}")
print()

# 风险提示
print("="*80)
print("风险提示")
print("="*80)
print()

for symbol, data in sentiments.items():
    if data['sentiment_score'] < -0.5:
        print(f"⚠️  {symbol}: 新闻情绪极度负面 ({data['sentiment_score']:+.3f})")
        print(f"   最新: {data['latest_headline']}")
        print(f"   建议: 考虑降低仓位或避开")
        print()
    elif data['sentiment_score'] > 0.5:
        print(f"✅ {symbol}: 新闻情绪积极 ({data['sentiment_score']:+.3f})")
        print(f"   最新: {data['latest_headline']}")
        print()

print("="*80)
