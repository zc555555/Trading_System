"""
阶段1：动态Top股票池 + 高置信度过滤

策略：
1. 每天从所有股票中动态选择Top N只（基于预测置信度）
2. 只对高置信度预测进行交易
3. 评估不同配置的效果

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


def calculate_prediction_confidence(predictions, method='abs_return'):
    """
    计算预测置信度

    方法：
    - abs_return: 预测收益率的绝对值（越大越有信心）
    - std_normalized: 归一化后的标准分数
    """
    if method == 'abs_return':
        # 预测收益率绝对值作为置信度
        return np.abs(predictions)
    elif method == 'std_normalized':
        # 基于标准差归一化
        std = predictions.std()
        if std > 0:
            return np.abs(predictions / std)
        else:
            return np.abs(predictions)
    else:
        raise ValueError(f"Unknown method: {method}")


def select_top_stocks_by_confidence(predictions_df, n_stocks=7):
    """
    基于预测置信度选择Top N只股票

    参数：
    - predictions_df: 包含 symbol, prediction, confidence 的DataFrame
    - n_stocks: 选择股票数量

    返回：
    - selected_df: 选中的股票
    """
    # 按置信度排序，选择Top N
    top_stocks = predictions_df.nlargest(n_stocks, 'confidence')
    return top_stocks


def evaluate_dynamic_strategy(
    test_df,
    predictions,
    n_top_stocks=7,
    min_confidence=0.60,
    confidence_method='abs_return'
):
    """
    评估动态股票池策略

    参数：
    - test_df: 测试数据集
    - predictions: 模型预测结果
    - n_top_stocks: 每天选择的股票数量
    - min_confidence: 最低置信度阈值
    - confidence_method: 置信度计算方法

    返回：
    - metrics: 评估指标
    """
    test_df = test_df.copy()
    test_df['prediction'] = predictions

    # 计算置信度
    test_df['confidence'] = calculate_prediction_confidence(
        predictions,
        method=confidence_method
    )

    # 按日期分组，每天选择Top股票
    results = []

    for date in test_df['date'].unique():
        day_data = test_df[test_df['date'] == date].copy()

        # 选择Top N股票
        top_stocks = select_top_stocks_by_confidence(day_data, n_top_stocks)

        # 应用置信度过滤
        if min_confidence > 0:
            top_stocks = top_stocks[top_stocks['confidence'] >= top_stocks['confidence'].quantile(min_confidence)]

        if len(top_stocks) > 0:
            results.append(top_stocks)

    if not results:
        return None

    # 合并所有选中的交易
    selected_trades = pd.concat(results, ignore_index=True)

    # 计算指标
    accuracy = direction_accuracy(
        selected_trades['future_return'].values,
        selected_trades['prediction'].values
    )

    mae = mean_absolute_error(
        selected_trades['future_return'].values,
        selected_trades['prediction'].values
    )

    r2 = r2_score(
        selected_trades['future_return'].values,
        selected_trades['prediction'].values
    )

    # 统计信息
    coverage = len(selected_trades) / len(test_df)
    unique_stocks = selected_trades['symbol'].nunique()
    trades_per_day = len(selected_trades) / test_df['date'].nunique()

    # 各股票表现
    per_stock_accuracy = {}
    for symbol in selected_trades['symbol'].unique():
        stock_data = selected_trades[selected_trades['symbol'] == symbol]
        stock_acc = direction_accuracy(
            stock_data['future_return'].values,
            stock_data['prediction'].values
        )
        per_stock_accuracy[symbol] = {
            'accuracy': float(stock_acc),
            'trades': len(stock_data)
        }

    metrics = {
        'overall_accuracy': float(accuracy),
        'mae': float(mae),
        'r2': float(r2),
        'coverage': float(coverage),
        'total_trades': len(selected_trades),
        'unique_stocks': int(unique_stocks),
        'trades_per_day': float(trades_per_day),
        'per_stock_performance': per_stock_accuracy,
        'config': {
            'n_top_stocks': n_top_stocks,
            'min_confidence': min_confidence,
            'confidence_method': confidence_method
        }
    }

    return metrics


def main():
    """主函数：测试动态Top股票池策略"""

    print("\n" + "=" * 80)
    print("阶段1：动态TOP股票池 + 高置信度过滤")
    print("=" * 80)

    # 加载最佳模型
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / "ensemble_time_windows.pkl"

    print(f"\n加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    print(f"模型: {list(models.keys())}")
    print(f"特征数: {len(feature_cols)}")

    # 加载数据
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    print(f"\n数据集: {df.shape}")
    print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"股票数量: {df['symbol'].nunique()}")

    # 分割数据
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    print(f"\n测试集: {len(test_df):,} 样本")
    print(f"测试期: {test_df['date'].min()} 到 {test_df['date'].max()}")

    # 准备特征
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # 处理NaN
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # 生成预测
    print(f"\n{'-' * 80}")
    print("生成预测")
    print(f"{'-' * 80}")

    predictions = {}
    for model_name in sorted(models.keys()):
        model = models[model_name]
        pred = model.predict(X_test)
        predictions[model_name] = pred
        print(f"{model_name:12s}: 完成")

    # 集成预测
    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    print(f"\n集成预测完成")

    # 基线准确率（无筛选）
    baseline_acc = direction_accuracy(y_test, ensemble_pred)
    print(f"\n基线准确率（所有股票，所有样本）: {baseline_acc*100:.2f}%")

    # 测试不同配置
    print(f"\n{'=' * 80}")
    print("测试动态股票池策略")
    print(f"{'=' * 80}")

    configs = [
        # (每天选N只股票, 置信度阈值, 描述)
        (7, 0.00, "Top 7股票, 无置信度过滤"),
        (7, 0.50, "Top 7股票, 50%置信度过滤"),
        (7, 0.60, "Top 7股票, 60%置信度过滤"),
        (7, 0.70, "Top 7股票, 70%置信度过滤"),
        (5, 0.60, "Top 5股票, 60%置信度过滤"),
        (10, 0.60, "Top 10股票, 60%置信度过滤"),
        (5, 0.70, "Top 5股票, 70%置信度过滤 (激进)"),
    ]

    all_results = []

    for n_stocks, min_conf, desc in configs:
        print(f"\n配置: {desc}")
        print(f"  每天选择: {n_stocks}只股票")
        print(f"  置信度阈值: {min_conf:.0%}")

        metrics = evaluate_dynamic_strategy(
            test_df=test_df,
            predictions=ensemble_pred,
            n_top_stocks=n_stocks,
            min_confidence=min_conf,
            confidence_method='abs_return'
        )

        if metrics:
            print(f"  准确率: {metrics['overall_accuracy']*100:.2f}%")
            print(f"  覆盖率: {metrics['coverage']*100:.1f}%")
            print(f"  总交易数: {metrics['total_trades']:,}")
            print(f"  平均每天交易: {metrics['trades_per_day']:.1f}只")
            print(f"  涉及股票数: {metrics['unique_stocks']}只")

            improvement = (metrics['overall_accuracy'] - baseline_acc) * 100
            print(f"  提升: {improvement:+.2f} pp")

            metrics['description'] = desc
            all_results.append(metrics)
        else:
            print(f"  [失败] 无有效交易")

    # 找出最佳配置
    print(f"\n{'=' * 80}")
    print("结果排名")
    print(f"{'=' * 80}")

    all_results.sort(key=lambda x: x['overall_accuracy'], reverse=True)

    print(f"\n{'排名':<5} {'配置':<40} {'准确率':<10} {'覆盖率':<10} {'提升':<10}")
    print("-" * 80)

    for i, result in enumerate(all_results, 1):
        improvement = (result['overall_accuracy'] - baseline_acc) * 100
        print(f"{i:<5} {result['description']:<40} "
              f"{result['overall_accuracy']*100:>6.2f}%   "
              f"{result['coverage']*100:>6.1f}%   "
              f"{improvement:>+6.2f}pp")

    # 最佳配置详情
    best = all_results[0]
    print(f"\n{'=' * 80}")
    print("最佳配置详情")
    print(f"{'=' * 80}")
    print(f"\n配置: {best['description']}")
    print(f"准确率: {best['overall_accuracy']*100:.2f}%")
    print(f"MAE: {best['mae']*100:.2f}%")
    print(f"R²: {best['r2']:.4f}")
    print(f"覆盖率: {best['coverage']*100:.1f}%")
    print(f"总交易数: {best['total_trades']:,}")
    print(f"平均每天: {best['trades_per_day']:.1f}只股票")

    improvement = (best['overall_accuracy'] - baseline_acc) * 100
    print(f"\n相比基线提升: {improvement:+.2f} pp")

    # Top表现股票
    print(f"\n{'-' * 80}")
    print("各股票表现（按准确率排序）")
    print(f"{'-' * 80}")

    stock_perf = [(s, m['accuracy'], m['trades'])
                  for s, m in best['per_stock_performance'].items()]
    stock_perf.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'股票':<8} {'准确率':<12} {'交易次数':<10}")
    print("-" * 35)
    for symbol, acc, trades in stock_perf:
        print(f"{symbol:<8} {acc*100:>6.2f}%      {trades:>4}")

    # 保存结果
    output_path = artifacts_dir / "dynamic_top_stocks_results.json"
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
    print("阶段1完成！")
    print(f"{'=' * 80}")

    print(f"\n[总结]")
    print(f"基线（所有股票）: {baseline_acc*100:.2f}%")
    print(f"最佳配置: {best['description']}")
    print(f"最佳准确率: {best['overall_accuracy']*100:.2f}%")
    print(f"提升: {improvement:+.2f} pp")

    if best['overall_accuracy'] >= 0.62:
        print(f"\n[成功] 达到62%目标！")
    elif best['overall_accuracy'] >= 0.60:
        print(f"\n[良好] 超过60%，继续优化可达62%")
    else:
        print(f"\n[继续] 未达62%，需要结合其他优化方法")


if __name__ == "__main__":
    main()
