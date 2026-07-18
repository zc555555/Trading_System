"""FinBERT情绪分析器 - 专门针对金融新闻"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用TensorFlow警告

try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("[WARNING] transformers not installed. Using simple sentiment analysis.")

# Simple sentiment fallback
def simple_analyze(text):
    """Simple keyword-based sentiment analysis"""
    positive_words = ['beat', 'surge', 'gain', 'profit', 'growth', 'strong', 'bullish', 'upgrade']
    negative_words = ['miss', 'fall', 'loss', 'weak', 'bearish', 'downgrade', 'concern', 'risk']

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return 0.5
    elif neg_count > pos_count:
        return -0.5
    return 0.0

class FinBERTSentimentAnalyzer:
    """使用FinBERT模型的情绪分析器"""

    def __init__(self):
        """初始化FinBERT模型"""
        self.analyzer = None

        if FINBERT_AVAILABLE:
            try:
                print("[INFO] Loading FinBERT model (first time will download ~500MB)...")
                # 使用ProsusAI/finbert模型，专门针对金融文本
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                print("[OK] FinBERT model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Failed to load FinBERT: {e}")
                print("[INFO] Falling back to simple sentiment analysis")
                self.analyzer = None
        else:
            print("[INFO] Using simple keyword-based sentiment analysis")

    def analyze(self, text: str) -> float:
        """
        分析文本情绪

        Args:
            text: 新闻标题或文本

        Returns:
            float: -1到1之间的情绪分数
                  -1 = 非常负面
                   0 = 中性
                  +1 = 非常正面
        """
        if not text:
            return 0.0

        # 如果FinBERT可用，使用它
        if self.analyzer is not None:
            try:
                # FinBERT返回: [{'label': 'positive/negative/neutral', 'score': 0.xx}]
                result = self.analyzer(text[:512])[0]  # FinBERT最多512个token

                label = result['label'].lower()
                confidence = result['score']

                # 转换为-1到1的分数
                if label == 'positive':
                    return confidence  # 0到1
                elif label == 'negative':
                    return -confidence  # -1到0
                else:  # neutral
                    return 0.0

            except Exception as e:
                print(f"[WARNING] FinBERT analysis failed: {e}, using fallback")
                return simple_analyze(text)

        # 否则使用简单的关键词分析
        return simple_analyze(text)

    def analyze_batch(self, texts: list) -> list:
        """
        批量分析多条文本

        Args:
            texts: 文本列表

        Returns:
            list: 情绪分数列表
        """
        if self.analyzer is not None and len(texts) > 0:
            try:
                # 批量处理更快
                results = self.analyzer([text[:512] for text in texts])

                scores = []
                for result in results:
                    label = result['label'].lower()
                    confidence = result['score']

                    if label == 'positive':
                        scores.append(confidence)
                    elif label == 'negative':
                        scores.append(-confidence)
                    else:
                        scores.append(0.0)

                return scores

            except Exception as e:
                print(f"[WARNING] Batch analysis failed: {e}, using fallback")
                return [simple_analyze(text) for text in texts]

        # 回退到简单分析
        return [simple_analyze(text) for text in texts]

    def get_batch_sentiment(self, symbols: list, delay: float = 0.5):
        """
        获取多个股票的新闻并分析情绪

        Args:
            symbols: 股票代码列表
            delay: 每次API调用之间的延迟（秒）

        Returns:
            dict: {symbol: {'sentiment_score': float, 'news_count': int}}
        """
        import time
        try:
            from config_finnhub import FINNHUB_API_KEY
            import finnhub
        except ImportError:
            print("[WARNING] Finnhub not configured, returning zero sentiment")
            return {symbol: {'sentiment_score': 0.0, 'news_count': 0} for symbol in symbols}

        # 初始化Finnhub客户端
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

        results = {}
        total = len(symbols)

        print(f"\nFetching news sentiment for {total} stocks...")
        print("="*60)

        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{total}] {symbol}")

            try:
                # 获取过去24小时的新闻
                from datetime import datetime, timedelta
                to_date = datetime.now()
                from_date = to_date - timedelta(hours=24)

                news = finnhub_client.company_news(
                    symbol,
                    _from=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d')
                )

                if not news:
                    print(f"  [INFO] {symbol}: 没有找到新闻")
                    results[symbol] = {'sentiment_score': 0.0, 'news_count': 0}
                    time.sleep(delay)
                    continue

                # 提取标题
                headlines = [item['headline'] for item in news if 'headline' in item]

                if not headlines:
                    results[symbol] = {'sentiment_score': 0.0, 'news_count': 0}
                    time.sleep(delay)
                    continue

                # 批量分析情绪
                sentiments = self.analyze_batch(headlines)
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

                results[symbol] = {
                    'sentiment_score': avg_sentiment,
                    'news_count': len(headlines)
                }

                # 显示结果
                sentiment_str = f"{avg_sentiment:+.4f}"
                trend = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
                print(f"  ✓ {symbol}: 情绪={sentiment_str}, 新闻数={len(headlines)}, 趋势={trend}")

            except Exception as e:
                print(f"  ✗ {symbol}: 获取失败 - {e}")
                results[symbol] = {'sentiment_score': 0.0, 'news_count': 0}

            # API限流延迟
            time.sleep(delay)

        print("="*60)
        print(f"完成！共获取 {total} 只股票的情绪数据\n")

        return results


# 全局单例
_finbert_analyzer = None

def get_finbert_analyzer():
    """获取全局FinBERT分析器实例"""
    global _finbert_analyzer
    if _finbert_analyzer is None:
        _finbert_analyzer = FinBERTSentimentAnalyzer()
    return _finbert_analyzer

def analyze_headline_finbert(headline: str) -> float:
    """
    使用FinBERT分析标题情绪

    Args:
        headline: 新闻标题

    Returns:
        float: -1到1之间的情绪分数
    """
    analyzer = get_finbert_analyzer()
    return analyzer.analyze(headline)


if __name__ == "__main__":
    # 测试
    print("Testing FinBERT Sentiment Analyzer...")
    print("="*60)

    analyzer = FinBERTSentimentAnalyzer()

    test_headlines = [
        "Apple stock surges on strong earnings beat, revenue jumps 20%",
        "Tesla shares plunge as production misses estimates badly",
        "Microsoft announces new AI breakthrough in natural language",
        "Amazon faces federal antitrust investigation over practices",
        "Google reports quarterly results in line with expectations",
        "NVIDIA rallies on optimistic AI chip demand forecast",
        "Meta warns of slowing ad revenue growth amid recession fears"
    ]

    print("\nSingle analysis:")
    for headline in test_headlines:
        score = analyzer.analyze(headline)

        if score > 0.3:
            emoji = "📈"
            sentiment = "Positive"
        elif score < -0.3:
            emoji = "📉"
            sentiment = "Negative"
        else:
            emoji = "➡️"
            sentiment = "Neutral"

        print(f"{score:+.3f} {emoji} {sentiment:<10} - {headline}")

    print("\n" + "="*60)
    print("\nBatch analysis:")
    scores = analyzer.analyze_batch(test_headlines)

    for headline, score in zip(test_headlines, scores):
        print(f"{score:+.3f} - {headline[:50]}...")

    print("\n" + "="*60)
    print("Test complete!")
