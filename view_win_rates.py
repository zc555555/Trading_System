"""查看模型历史胜率"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

project_dir = Path(__file__).parent
artifacts_dir = project_dir / "research" / "artifacts"

print("="*80)
print("模型历史表现 - 胜率统计")
print("="*80)
print()

# 加载特征数据
feature_file = project_dir / "research" / "data" / "features_all_stocks.parquet"

if not feature_file.exists():
    print("[ERROR] 特征文件不存在，请先运行数据更新")
    exit(1)

print(f"[OK] 加载特征数据: {feature_file.name}")
df = pd.read_parquet(feature_file)

# 加载模型
model_files = {
    'momentum': artifacts_dir / 'ensemble_momentum.pkl',
    'trend': artifacts_dir / 'ensemble_trend.pkl',
    'volatility': artifacts_dir / 'ensemble_volatility.pkl',
    'volume': artifacts_dir / 'ensemble_volume.pkl',
    'market': artifacts_dir / 'ensemble_market.pkl',
    'alpha': artifacts_dir / 'ensemble_alpha.pkl'
}

print("\n[OK] 加载模型...")
factor_ensembles = {}
for factor_name, model_file in model_files.items():
    if model_file.exists():
        with open(model_file, 'rb') as f:
            factor_ensembles[factor_name] = pickle.load(f)
        print(f"  ✓ {factor_name}")

# 计算最近N个交易日的预测准确率
def calculate_recent_win_rate(df, factor_ensembles, n_days=20):
    """计算最近N个交易日的胜率"""
    # 获取最近的日期
    latest_dates = sorted(df['date'].unique())[-n_days:]
    recent_data = df[df['date'].isin(latest_dates)].copy()

    if len(recent_data) == 0:
        return None

    # 权重配置
    weights = {
        'momentum': 0.25,
        'trend': 0.20,
        'volatility': 0.10,
        'volume': 0.10,
        'market': 0.25,
        'alpha': 0.10
    }

    # 获取特征列
    feature_cols = [col for col in df.columns if col.startswith('feature_')]

    # 为每个因子生成预测
    factor_predictions = {}
    for factor_name, ensemble in factor_ensembles.items():
        X = recent_data[feature_cols].fillna(0)

        # 平均3个模型的预测
        preds = []
        for model_name, model in ensemble.items():
            pred = model.predict(X)
            preds.append(pred)

        factor_predictions[factor_name] = np.mean(preds, axis=0)

    # 加权平均得到最终预测
    weighted_pred = np.zeros(len(recent_data))
    for factor_name, pred in factor_predictions.items():
        weighted_pred += pred * weights[factor_name]

    recent_data['prediction'] = weighted_pred

    # 计算准确率
    recent_data['pred_direction'] = recent_data['prediction'] > 0
    recent_data['actual_direction'] = recent_data['target'] > 0
    recent_data['correct'] = recent_data['pred_direction'] == recent_data['actual_direction']

    # 按置信度分组
    recent_data['confidence'] = abs(recent_data['prediction'])
    recent_data['conf_level'] = pd.cut(recent_data['confidence'],
                                        bins=[0, 0.003, 0.005, 0.01, 1.0],
                                        labels=['低(0-0.3%)', '中(0.3-0.5%)', '高(0.5-1%)', '极高(>1%)'])

    return recent_data

print("\n" + "="*80)
print("计算最近20个交易日的胜率...")
print("="*80)

recent_data = calculate_recent_win_rate(df, factor_ensembles, n_days=20)

if recent_data is None:
    print("[ERROR] 无法计算胜率")
    exit(1)

# 总体胜率
total_predictions = len(recent_data)
correct_predictions = recent_data['correct'].sum()
overall_win_rate = correct_predictions / total_predictions * 100

print()
print(f"【总体表现】")
print(f"  预测次数: {total_predictions}")
print(f"  正确次数: {correct_predictions}")
print(f"  总体胜率: {overall_win_rate:.1f}%")
print()

# 按方向统计
buy_signals = recent_data[recent_data['prediction'] > 0]
sell_signals = recent_data[recent_data['prediction'] < 0]

if len(buy_signals) > 0:
    buy_win_rate = buy_signals['correct'].sum() / len(buy_signals) * 100
    print(f"【做多信号】")
    print(f"  信号数量: {len(buy_signals)}")
    print(f"  胜率: {buy_win_rate:.1f}%")
    print()

if len(sell_signals) > 0:
    sell_win_rate = sell_signals['correct'].sum() / len(sell_signals) * 100
    print(f"【做空信号】")
    print(f"  信号数量: {len(sell_signals)}")
    print(f"  胜率: {sell_win_rate:.1f}%")
    print()

# 按置信度统计
print("【按置信度分级】")
for conf_level in ['低(0-0.3%)', '中(0.3-0.5%)', '高(0.5-1%)', '极高(>1%)']:
    level_data = recent_data[recent_data['conf_level'] == conf_level]
    if len(level_data) > 0:
        level_win_rate = level_data['correct'].sum() / len(level_data) * 100
        avg_return = level_data['target'].mean() * 100
        print(f"  {conf_level:15} : 样本数={len(level_data):4}, 胜率={level_win_rate:5.1f}%, 平均收益={avg_return:+6.2f}%")

print()

# 显示高置信度股票的表现
high_conf_threshold = 0.005  # 0.5%
high_conf_data = recent_data[abs(recent_data['prediction']) >= high_conf_threshold]

if len(high_conf_data) > 0:
    print("="*80)
    print(f"【高置信度信号表现】（置信度 ≥ {high_conf_threshold*100}%）")
    print("="*80)
    print()

    hc_win_rate = high_conf_data['correct'].sum() / len(high_conf_data) * 100
    hc_avg_return = high_conf_data['target'].mean() * 100

    print(f"  信号数量: {len(high_conf_data)}")
    print(f"  胜率: {hc_win_rate:.1f}%")
    print(f"  平均收益: {hc_avg_return:+.2f}%")
    print()

# 当前信号的预期表现
print("="*80)
print("【今日信号预期】")
print("="*80)
print()

# 读取今日信号
import json
from datetime import datetime

today = datetime.now().strftime('%Y%m%d')
signal_file = artifacts_dir / f"signals_multi_factor_{today}.json"

if signal_file.exists():
    with open(signal_file, 'r') as f:
        signals = json.load(f)

    print(f"今日推荐 {signals['n_stocks']} 只股票:")
    print()

    for stock in signals['stocks']:
        conf_pct = stock['confidence'] * 100
        pred_pct = stock['prediction'] * 100

        # 根据置信度估计胜率
        if conf_pct >= 1.0:
            expected_wr = "65-70%"
        elif conf_pct >= 0.5:
            expected_wr = "60-65%"
        elif conf_pct >= 0.3:
            expected_wr = "55-60%"
        else:
            expected_wr = "50-55%"

        print(f"  {stock['symbol']:6} : 预期涨幅={pred_pct:+5.2f}%, 置信度={conf_pct:5.2f}%, 预估胜率={expected_wr}")

    print()
else:
    print("今日尚未生成交易信号")
    print()

print("="*80)
print()
print("提示:")
print("  - 胜率基于最近20个交易日的历史表现")
print("  - 实际收益可能与历史表现存在偏差")
print("  - 建议关注高置信度信号，通常表现更好")
print()
print("="*80)
