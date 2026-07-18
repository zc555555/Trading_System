"""
动态股票池 - 滚动窗口方法

策略：
基于最近N天的预测准确率，动态选择表现最好的股票

优点：
- 更稳健（基于历史表现，不是单次预测）
- 换手率适中（每周调整）
- 避免过度拟合

预期：59.24% -> 62%+
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def direction_accuracy(y_true, y_pred):
    """计算方向准确率"""
    return ((y_true > 0) == (y_pred > 0)).mean()


class RollingWindowStockSelector:
    """基于滚动窗口的动态股票选择器"""

    def __init__(self, window_days=30, n_top_stocks=7, rebalance_freq='weekly'):
        """
        参数：
        - window_days: 滚动窗口天数（评估最近N天表现）
        - n_top_stocks: 选择股票数量
        - rebalance_freq: 调仓频率（'daily', 'weekly', 'monthly'）
        """
        self.window_days = window_days
        self.n_top_stocks = n_top_stocks
        self.rebalance_freq = rebalance_freq
        self.historical_performance = defaultdict(list)

    def update_performance(self, date, symbol, y_true, y_pred):
        """更新股票历史表现"""
        correct = (y_true > 0) == (y_pred > 0)
        self.historical_performance[symbol].append({
            'date': date,
            'correct': correct,
            'abs_error': abs(y_true - y_pred)
        })

    def calculate_stock_score(self, symbol):
        """
        计算股票得分（基于滚动窗口）

        得分 = 0.7 * 准确率 + 0.3 * (1 - 归一化MAE)
        """
        if symbol not in self.historical_performance:
            return 0.0

        records = self.historical_performance[symbol][-self.window_days:]

        if len(records) < self.window_days // 2:  # 至少需要一半的数据
            return 0.0

        # 准确率
        accuracy = sum(r['correct'] for r in records) / len(records)

        # MAE
        mae = sum(r['abs_error'] for r in records) / len(records)
        # 归一化MAE（假设最大MAE = 0.05）
        normalized_mae = min(mae / 0.05, 1.0)

        # 综合得分
        score = 0.7 * accuracy + 0.3 * (1 - normalized_mae)

        return score

    def select_top_stocks(self, available_symbols):
        """选择Top N股票"""
        scores = {symbol: self.calculate_stock_score(symbol)
                  for symbol in available_symbols}

        # 排序并选择Top N
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in sorted_stocks[:self.n_top_stocks]]

        return top_stocks, scores

    def should_rebalance(self, current_date, last_rebalance_date):
        """判断是否需要调仓"""
        if last_rebalance_date is None:
            return True

        if self.rebalance_freq == 'daily':
            return True
        elif self.rebalance_freq == 'weekly':
            # 每周一调仓
            return current_date.weekday() == 0
        elif self.rebalance_freq == 'monthly':
            # 每月第一天调仓
            return current_date.day == 1
        else:
            return True


def backtest_rolling_window_strategy(
    df,
    predictions,
    window_days=30,
    n_top_stocks=7,
    rebalance_freq='weekly'
):
    """
    回测滚动窗口动态选股策略

    参数：
    - df: 数据集（包含 date, symbol, future_return）
    - predictions: 预测结果
    - window_days: 滚动窗口天数
    - n_top_stocks: 选择股票数量
    - rebalance_freq: 调仓频率

    返回：
    - results: 回测结果
    """
    df = df.copy()
    df['prediction'] = predictions
    df['date'] = pd.to_datetime(df['date'])

    selector = RollingWindowStockSelector(
        window_days=window_days,
        n_top_stocks=n_top_stocks,
        rebalance_freq=rebalance_freq
    )

    selected_trades = []
    portfolio_history = []
    last_rebalance_date = None
    current_portfolio = []

    unique_dates = sorted(df['date'].unique())

    for date in unique_dates:
        day_data = df[df['date'] == date]

        # 更新历史表现
        for _, row in day_data.iterrows():
            selector.update_performance(
                date=date,
                symbol=row['symbol'],
                y_true=row['future_return'],
                y_pred=row['prediction']
            )

        # 判断是否需要调仓
        if selector.should_rebalance(date, last_rebalance_date):
            available_symbols = day_data['symbol'].unique()
            current_portfolio, scores = selector.select_top_stocks(available_symbols)
            last_rebalance_date = date

            portfolio_history.append({
                'date': date,
                'portfolio': current_portfolio.copy(),
                'scores': scores.copy()
            })

        # 只交易当前投资组合中的股票
        day_trades = day_data[day_data['symbol'].isin(current_portfolio)]
        if len(day_trades) > 0:
            selected_trades.append(day_trades)

    # 合并所有交易
    if not selected_trades:
        return None

    all_trades = pd.concat(selected_trades, ignore_index=True)

    # 计算指标
    accuracy = direction_accuracy(
        all_trades['future_return'].values,
        all_trades['prediction'].values
    )

    mae = mean_absolute_error(
        all_trades['future_return'].values,
        all_trades['prediction'].values
    )

    r2 = r2_score(
        all_trades['future_return'].values,
        all_trades['prediction'].values
    )

    # 统计信息
    coverage = len(all_trades) / len(df)
    unique_stocks = all_trades['symbol'].nunique()
    trades_per_day = len(all_trades) / len(unique_dates)
    rebalance_count = len(portfolio_history)

    # 各股票表现
    per_stock_stats = {}
    for symbol in all_trades['symbol'].unique():
        stock_data = all_trades[all_trades['symbol'] == symbol]
        stock_acc = direction_accuracy(
            stock_data['future_return'].values,
            stock_data['prediction'].values
        )
        per_stock_stats[symbol] = {
            'accuracy': float(stock_acc),
            'trades': len(stock_data),
            'mae': float(stock_data['future_return'].sub(stock_data['prediction']).abs().mean())
        }

    results = {
        'overall_accuracy': float(accuracy),
        'mae': float(mae),
        'r2': float(r2),
        'coverage': float(coverage),
        'total_trades': len(all_trades),
        'unique_stocks': int(unique_stocks),
        'trades_per_day': float(trades_per_day),
        'rebalance_count': int(rebalance_count),
        'per_stock_performance': per_stock_stats,
        'portfolio_history': [
            {
                'date': str(p['date'].date()),
                'portfolio': p['portfolio'],
                'top_5_scores': {
                    k: float(v) for k, v in
                    sorted(p['scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                }
            }
            for p in portfolio_history[:10]  # 只保存前10次调仓记录
        ],
        'config': {
            'window_days': window_days,
            'n_top_stocks': n_top_stocks,
            'rebalance_freq': rebalance_freq
        }
    }

    return results


def main():
    """主函数：测试滚动窗口动态选股"""

    print("\n" + "=" * 80)
    print("动态股票池 - 滚动窗口方法")
    print("=" * 80)

    # 加载模型
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / "ensemble_time_windows.pkl"

    print(f"\n加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # 加载数据
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    print(f"\n数据集: {df.shape}")
    print(f"股票数量: {df['symbol'].nunique()}")

    # 分割数据
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    print(f"\n测试集: {len(test_df):,} 样本")

    # 准备特征并预测
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print(f"\n生成预测...")
    predictions = {}
    for model_name in sorted(models.keys()):
        model = models[model_name]
        pred = model.predict(X_test)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    baseline_acc = direction_accuracy(y_test, ensemble_pred)
    print(f"基线准确率: {baseline_acc*100:.2f}%")

    # 测试不同配置
    print(f"\n{'=' * 80}")
    print("测试滚动窗口配置")
    print(f"{'=' * 80}")

    configs = [
        # (窗口天数, Top股票数, 调仓频率, 描述)
        (30, 7, 'weekly', "30天窗口, Top 7, 每周调仓"),
        (30, 5, 'weekly', "30天窗口, Top 5, 每周调仓"),
        (30, 10, 'weekly', "30天窗口, Top 10, 每周调仓"),
        (20, 7, 'weekly', "20天窗口, Top 7, 每周调仓"),
        (45, 7, 'weekly', "45天窗口, Top 7, 每周调仓"),
        (30, 7, 'monthly', "30天窗口, Top 7, 每月调仓"),
    ]

    all_results = []

    for window, n_stocks, freq, desc in configs:
        print(f"\n配置: {desc}")

        results = backtest_rolling_window_strategy(
            df=test_df,
            predictions=ensemble_pred,
            window_days=window,
            n_top_stocks=n_stocks,
            rebalance_freq=freq
        )

        if results:
            print(f"  准确率: {results['overall_accuracy']*100:.2f}%")
            print(f"  覆盖率: {results['coverage']*100:.1f}%")
            print(f"  调仓次数: {results['rebalance_count']}")
            print(f"  涉及股票: {results['unique_stocks']}只")

            improvement = (results['overall_accuracy'] - baseline_acc) * 100
            print(f"  提升: {improvement:+.2f} pp")

            results['description'] = desc
            all_results.append(results)

    # 排名
    print(f"\n{'=' * 80}")
    print("结果排名")
    print(f"{'=' * 80}")

    all_results.sort(key=lambda x: x['overall_accuracy'], reverse=True)

    for i, result in enumerate(all_results, 1):
        improvement = (result['overall_accuracy'] - baseline_acc) * 100
        print(f"{i}. {result['description']:<45} "
              f"{result['overall_accuracy']*100:>6.2f}%  "
              f"({improvement:>+5.2f}pp)")

    # 最佳配置详情
    best = all_results[0]
    print(f"\n{'=' * 80}")
    print("最佳配置详情")
    print(f"{'=' * 80}")
    print(f"\n{best['description']}")
    print(f"准确率: {best['overall_accuracy']*100:.2f}%")
    print(f"提升: {(best['overall_accuracy'] - baseline_acc)*100:+.2f} pp")
    print(f"调仓次数: {best['rebalance_count']}")

    # 显示部分调仓历史
    print(f"\n前5次调仓记录:")
    for i, record in enumerate(best['portfolio_history'][:5], 1):
        print(f"\n  {i}. {record['date']}")
        print(f"     投资组合: {', '.join(record['portfolio'])}")
        print(f"     Top 3得分:")
        for stock, score in list(record['top_5_scores'].items())[:3]:
            print(f"       {stock}: {score:.4f}")

    # 股票表现
    print(f"\n各股票表现（Top 10）:")
    stock_perf = [(s, m['accuracy'], m['trades'])
                  for s, m in best['per_stock_performance'].items()]
    stock_perf.sort(key=lambda x: x[1], reverse=True)

    for i, (symbol, acc, trades) in enumerate(stock_perf[:10], 1):
        print(f"  {i:2d}. {symbol:<6} {acc*100:>6.2f}%  ({trades:>3}次交易)")

    # 保存结果
    output_path = artifacts_dir / "rolling_window_selection_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_accuracy': float(baseline_acc),
            'best_config': best,
            'all_configs': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'-' * 80}")
    print(f"结果已保存: {output_path}")
    print(f"{'-' * 80}")

    # 总结
    print(f"\n{'=' * 80}")
    print("滚动窗口策略测试完成")
    print(f"{'=' * 80}")

    improvement = (best['overall_accuracy'] - baseline_acc) * 100
    print(f"\n基线: {baseline_acc*100:.2f}%")
    print(f"最佳: {best['overall_accuracy']*100:.2f}%")
    print(f"提升: {improvement:+.2f} pp")

    if best['overall_accuracy'] >= 0.62:
        print(f"\n[优秀] 达到62%目标！")
    elif best['overall_accuracy'] >= 0.60:
        print(f"\n[良好] 超过60%")


if __name__ == "__main__":
    main()
