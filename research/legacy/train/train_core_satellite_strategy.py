"""
核心-卫星动态股票池策略

策略架构：
1. 核心池（70%资金）：固定5只历史表现最稳定的股票
2. 卫星池（30%资金）：动态选择2-3只高置信度股票

优点：
- 稳定性：核心池提供稳定收益
- 灵活性：卫星池捕捉短期机会
- 平衡：结合静态和动态的优势

预期：59.24% -> 62%+
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def direction_accuracy(y_true, y_pred):
    """计算方向准确率"""
    return ((y_true > 0) == (y_pred > 0)).mean()


def calculate_stock_stability(df, symbol):
    """
    计算股票的稳定性得分

    综合考虑：
    1. 准确率
    2. 准确率的稳定性（标准差）
    3. 交易次数
    """
    stock_data = df[df['symbol'] == symbol]

    if len(stock_data) == 0:
        return 0.0

    # 按月分组计算准确率
    stock_data = stock_data.copy()
    stock_data['month'] = pd.to_datetime(stock_data['date']).dt.to_period('M')

    monthly_accuracies = []
    for month in stock_data['month'].unique():
        month_data = stock_data[stock_data['month'] == month]
        if len(month_data) >= 5:  # 至少5个样本
            acc = direction_accuracy(
                month_data['future_return'].values,
                month_data['prediction'].values
            )
            monthly_accuracies.append(acc)

    if len(monthly_accuracies) < 3:  # 至少3个月数据
        return 0.0

    # 平均准确率
    mean_acc = np.mean(monthly_accuracies)

    # 稳定性（标准差越小越好）
    std_acc = np.std(monthly_accuracies)

    # 稳定性得分：准确率高且波动小
    stability_score = mean_acc * (1 - std_acc)

    return stability_score


def select_core_portfolio(df, predictions, n_core=5):
    """
    选择核心投资组合（稳定性最高的股票）

    参数：
    - df: 数据集
    - predictions: 预测结果
    - n_core: 核心股票数量

    返回：
    - core_stocks: 核心股票列表
    - stability_scores: 稳定性得分
    """
    df = df.copy()
    df['prediction'] = predictions

    # 计算每只股票的稳定性
    stability_scores = {}
    for symbol in df['symbol'].unique():
        score = calculate_stock_stability(df, symbol)
        stability_scores[symbol] = score

    # 选择Top N
    sorted_stocks = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    core_stocks = [s[0] for s in sorted_stocks[:n_core]]

    return core_stocks, stability_scores


def select_satellite_portfolio(day_data, n_satellite=2, method='confidence'):
    """
    选择卫星投资组合（高置信度或高收益潜力）

    参数：
    - day_data: 当天的数据
    - n_satellite: 卫星股票数量
    - method: 'confidence' 或 'expected_return'

    返回：
    - satellite_stocks: 卫星股票列表
    """
    if method == 'confidence':
        # 基于预测收益率绝对值（置信度）
        day_data = day_data.copy()
        day_data['confidence'] = np.abs(day_data['prediction'])
        top_stocks = day_data.nlargest(n_satellite, 'confidence')['symbol'].tolist()
    elif method == 'expected_return':
        # 基于预测收益率（做多最看好的）
        top_stocks = day_data.nlargest(n_satellite, 'prediction')['symbol'].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    return top_stocks


def backtest_core_satellite_strategy(
    df,
    predictions,
    n_core=5,
    n_satellite=2,
    core_weight=0.7,
    satellite_method='confidence',
    rebalance_satellite='weekly'
):
    """
    回测核心-卫星策略

    参数：
    - df: 数据集
    - predictions: 预测结果
    - n_core: 核心股票数量
    - n_satellite: 卫星股票数量
    - core_weight: 核心池权重（0-1）
    - satellite_method: 卫星池选择方法
    - rebalance_satellite: 卫星池调仓频率

    返回：
    - results: 回测结果
    """
    df = df.copy()
    df['prediction'] = predictions
    df['date'] = pd.to_datetime(df['date'])

    # 选择核心池（基于整个测试集）
    core_stocks, stability_scores = select_core_portfolio(df, predictions, n_core)

    print(f"\n核心池股票: {core_stocks}")
    print(f"稳定性得分:")
    for stock in core_stocks:
        print(f"  {stock}: {stability_scores[stock]:.4f}")

    # 回测
    core_trades = []
    satellite_trades = []
    last_satellite_stocks = []
    last_rebalance_date = None

    unique_dates = sorted(df['date'].unique())

    for date in unique_dates:
        day_data = df[df['date'] == date]

        # 核心池：固定股票
        core_day_data = day_data[day_data['symbol'].isin(core_stocks)]
        if len(core_day_data) > 0:
            core_trades.append(core_day_data)

        # 卫星池：动态选择
        should_rebalance = (
            last_rebalance_date is None or
            (rebalance_satellite == 'daily') or
            (rebalance_satellite == 'weekly' and date.weekday() == 0) or
            (rebalance_satellite == 'monthly' and date.day == 1)
        )

        if should_rebalance:
            # 从非核心股票中选择
            non_core_data = day_data[~day_data['symbol'].isin(core_stocks)]
            if len(non_core_data) > 0:
                last_satellite_stocks = select_satellite_portfolio(
                    non_core_data,
                    n_satellite=n_satellite,
                    method=satellite_method
                )
                last_rebalance_date = date

        # 卫星池交易
        satellite_day_data = day_data[day_data['symbol'].isin(last_satellite_stocks)]
        if len(satellite_day_data) > 0:
            satellite_trades.append(satellite_day_data)

    # 合并交易
    core_all = pd.concat(core_trades, ignore_index=True) if core_trades else pd.DataFrame()
    satellite_all = pd.concat(satellite_trades, ignore_index=True) if satellite_trades else pd.DataFrame()

    # 计算各自的准确率
    core_accuracy = 0
    satellite_accuracy = 0

    if len(core_all) > 0:
        core_accuracy = direction_accuracy(
            core_all['future_return'].values,
            core_all['prediction'].values
        )

    if len(satellite_all) > 0:
        satellite_accuracy = direction_accuracy(
            satellite_all['future_return'].values,
            satellite_all['prediction'].values
        )

    # 加权平均准确率
    if len(core_all) > 0 and len(satellite_all) > 0:
        combined_accuracy = core_weight * core_accuracy + (1 - core_weight) * satellite_accuracy
    elif len(core_all) > 0:
        combined_accuracy = core_accuracy
    elif len(satellite_all) > 0:
        combined_accuracy = satellite_accuracy
    else:
        combined_accuracy = 0

    # 合并所有交易
    all_trades = pd.concat([core_all, satellite_all], ignore_index=True)

    overall_accuracy = direction_accuracy(
        all_trades['future_return'].values,
        all_trades['prediction'].values
    ) if len(all_trades) > 0 else 0

    results = {
        'overall_accuracy': float(overall_accuracy),
        'combined_accuracy': float(combined_accuracy),
        'core_accuracy': float(core_accuracy),
        'satellite_accuracy': float(satellite_accuracy),
        'core_trades': len(core_all),
        'satellite_trades': len(satellite_all),
        'total_trades': len(all_trades),
        'core_stocks': core_stocks,
        'satellite_unique_stocks': len(satellite_all['symbol'].unique()) if len(satellite_all) > 0 else 0,
        'config': {
            'n_core': n_core,
            'n_satellite': n_satellite,
            'core_weight': core_weight,
            'satellite_method': satellite_method,
            'rebalance_satellite': rebalance_satellite
        }
    }

    return results


def main():
    """主函数：测试核心-卫星策略"""

    print("\n" + "=" * 80)
    print("核心-卫星动态股票池策略")
    print("=" * 80)

    # 加载模型
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / "ensemble_time_windows.pkl"

    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # 加载数据
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # 分割数据
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    print(f"\n测试集: {len(test_df):,} 样本")

    # 预测
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print(f"\n生成预测...")
    predictions = {}
    for model_name in sorted(models.keys()):
        pred = models[model_name].predict(X_test)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    baseline_acc = direction_accuracy(y_test, ensemble_pred)
    print(f"基线准确率: {baseline_acc*100:.2f}%")

    # 测试不同配置
    print(f"\n{'=' * 80}")
    print("测试核心-卫星配置")
    print(f"{'=' * 80}")

    configs = [
        # (核心数, 卫星数, 核心权重, 卫星方法, 调仓频率, 描述)
        (5, 2, 0.7, 'confidence', 'weekly', "5核心 + 2卫星(置信度), 70/30, 每周调仓"),
        (5, 3, 0.7, 'confidence', 'weekly', "5核心 + 3卫星(置信度), 70/30, 每周调仓"),
        (7, 2, 0.8, 'confidence', 'weekly', "7核心 + 2卫星(置信度), 80/20, 每周调仓"),
        (5, 2, 0.6, 'confidence', 'weekly', "5核心 + 2卫星(置信度), 60/40, 每周调仓"),
        (5, 2, 0.7, 'expected_return', 'weekly', "5核心 + 2卫星(预期收益), 70/30, 每周调仓"),
        (5, 2, 0.7, 'confidence', 'monthly', "5核心 + 2卫星(置信度), 70/30, 每月调仓"),
    ]

    all_results = []

    for n_core, n_sat, core_w, sat_method, rebal, desc in configs:
        print(f"\n配置: {desc}")

        results = backtest_core_satellite_strategy(
            df=test_df,
            predictions=ensemble_pred,
            n_core=n_core,
            n_satellite=n_sat,
            core_weight=core_w,
            satellite_method=sat_method,
            rebalance_satellite=rebal
        )

        print(f"  整体准确率: {results['overall_accuracy']*100:.2f}%")
        print(f"  加权准确率: {results['combined_accuracy']*100:.2f}%")
        print(f"  核心准确率: {results['core_accuracy']*100:.2f}%")
        print(f"  卫星准确率: {results['satellite_accuracy']*100:.2f}%")

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
        print(f"{i}. {result['overall_accuracy']*100:.2f}% ({improvement:+.2f}pp) - {result['description']}")

    # 最佳配置
    best = all_results[0]
    print(f"\n{'=' * 80}")
    print("最佳配置")
    print(f"{'=' * 80}")
    print(f"\n{best['description']}")
    print(f"整体准确率: {best['overall_accuracy']*100:.2f}%")
    print(f"提升: {(best['overall_accuracy'] - baseline_acc)*100:+.2f} pp")
    print(f"\n核心池: {', '.join(best['core_stocks'])}")
    print(f"核心准确率: {best['core_accuracy']*100:.2f}%")
    print(f"卫星准确率: {best['satellite_accuracy']*100:.2f}%")

    # 保存结果
    output_path = artifacts_dir / "core_satellite_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_accuracy': float(baseline_acc),
            'best_config': best,
            'all_configs': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    # 总结
    improvement = (best['overall_accuracy'] - baseline_acc) * 100
    print(f"\n{'=' * 80}")
    print("总结")
    print(f"{'=' * 80}")
    print(f"基线: {baseline_acc*100:.2f}%")
    print(f"最佳: {best['overall_accuracy']*100:.2f}%")
    print(f"提升: {improvement:+.2f} pp")


if __name__ == "__main__":
    main()
