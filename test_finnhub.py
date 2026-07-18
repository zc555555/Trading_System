"""测试Finnhub连接和新闻获取"""
from news_sentiment_analyzer import NewsSentimentAnalyzer

print("="*80)
print("Finnhub新闻情绪分析测试")
print("="*80)
print()

# 初始化分析器
analyzer = NewsSentimentAnalyzer()

# 测试股票（包括英伟达）
test_symbols = ['NVDA', 'AAPL', 'MSFT', 'TSLA']

print(f"测试 {len(test_symbols)} 只股票的实时新闻情绪...")
print()

# 获取情绪
sentiments = analyzer.get_batch_sentiment(test_symbols, delay=1.0)

# 显示摘要
analyzer.display_summary(sentiments)

# 显示NVDA的详细新闻
print("\n" + "="*80)
print("NVDA 最新新闻详情")
print("="*80)
print()

nvda_data = sentiments.get('NVDA', {})

if nvda_data.get('news_summary'):
    for i, news in enumerate(nvda_data['news_summary'], 1):
        print(f"{i}. [{news['source']}]")
        print(f"   标题: {news['headline']}")
        print(f"   情绪: {news['sentiment']:+.3f}")
        print(f"   时间: {news['time']}")
        print()
else:
    print("没有找到NVDA的新闻")

print("="*80)
print()
print("测试完成！")
print()
print("如果看到新闻数据，说明Finnhub连接成功！")
print()
