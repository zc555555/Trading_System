"""深度分析一个月的交易数据 - 为模型精调提供依据"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

project_dir = Path(__file__).parent
logs_dir = project_dir / "trading_logs"

print("="*80)
print("完整交易数据深度分析")
print("="*80)
print()

# 读取所有日志文件
signal_log = logs_dir / "daily_signals_detailed.csv"
config_log = logs_dir / "daily_config.csv"
account_log = logs_dir / "daily_account.csv"
position_log = logs_dir / "daily_positions.csv"

if not signal_log.exists():
    print("[ERROR] 交易信号日志不存在")
    print("请先运行 log_comprehensive_trading.py 收集数据")
    exit(1)

# 加载数据
signals_df = pd.read_csv(signal_log)
signals_df['date'] = pd.to_datetime(signals_df['date'])

if config_log.exists():
    config_df = pd.read_csv(config_log)
    config_df['date'] = pd.to_datetime(config_df['date'])
else:
    config_df = None

if account_log.exists():
    account_df = pd.read_csv(account_log)
    account_df['date'] = pd.to_datetime(account_df['date'])
else:
    account_df = None

trading_days = signals_df['date'].nunique()
total_trades = len(signals_df)

print(f"[OK] 数据加载完成")
print(f"  交易日数: {trading_days}")
print(f"  总交易数: {total_trades}")
print(f"  日期范围: {signals_df['date'].min().date()} ~ {signals_df['date'].max().date()}")
print()

if trading_days < 10:
    print("⚠️  数据不足（少于10个交易日）")
    print("   建议收集至少20个交易日的数据")
    print()

# ============================================================================
# 1. 模型预测准确性分析
# ============================================================================

print("="*80)
print("【1. 模型预测准确性】")
print("="*80)
print()

if 'actual_return' in signals_df.columns:
    completed = signals_df[signals_df['actual_return'].notna()].copy()

    if len(completed) > 0:
        # 方向准确率
        direction_correct = (completed['prediction_direction_correct'] == True).sum()
        direction_accuracy = direction_correct / len(completed) * 100

        print(f"方向预测准确率: {direction_accuracy:.1f}%")
        print()

        # 预测误差分析
        pred_error_mean = completed['prediction_error'].mean()
        pred_error_std = completed['prediction_error'].std()
        pred_error_mae = completed['prediction_error'].abs().mean()

        print(f"预测误差统计:")
        print(f"  平均误差 (bias):      {pred_error_mean:+.3f}%")
        print(f"  误差标准差 (std):     {pred_error_std:.3f}%")
        print(f"  平均绝对误差 (MAE):   {pred_error_mae:.3f}%")
        print()

        if abs(pred_error_mean) > 0.1:
            if pred_error_mean > 0:
                print(f"  ⚠️  模型系统性**低估**收益（过于保守）")
            else:
                print(f"  ⚠️  模型系统性**高估**收益（过于乐观）")
            print()

        # 实际表现
        avg_prediction = completed['prediction_pct'].mean()
        avg_expected = completed['expected_return'].mean()
        avg_actual = completed['actual_return'].mean()

        print(f"收益对比:")
        print(f"  模型预测平均: {avg_prediction:+.3f}%")
        print(f"  预期收益平均: {avg_expected:+.3f}%  (收盘→收盘)")
        print(f"  实际收益平均: {avg_actual:+.3f}%  (开盘→收盘)")
        print()

        # 胜率
        win_rate = (completed['trade_profitable'] == True).sum() / len(completed) * 100
        print(f"实际交易胜率: {win_rate:.1f}%")
        print()
else:
    print("[INFO] 尚无完整的实际表现数据")
    print()

# ============================================================================
# 2. 因子有效性分析
# ============================================================================

print("="*80)
print("【2. 各因子有效性分析】")
print("="*80)
print()

if 'actual_return' in signals_df.columns:
    completed = signals_df[signals_df['actual_return'].notna()].copy()

    if len(completed) > 0:
        factors = ['momentum', 'trend', 'volatility', 'volume', 'market', 'alpha']

        print("各因子贡献与实际收益的相关性:")
        print()
        print("  因子           相关系数    平均贡献   权重    有效性评级")
        print("-" * 70)

        factor_analysis = []

        for factor in factors:
            factor_col = f'factor_{factor}_pct'
            if factor_col in completed.columns:
                correlation = completed[factor_col].corr(completed['actual_return'])
                avg_contribution = completed[factor_col].mean()

                # 从config获取权重
                if config_df is not None and len(config_df) > 0:
                    weight = config_df[f'weight_{factor}'].iloc[-1]
                else:
                    weight = np.nan

                # 有效性评级
                if abs(correlation) > 0.3:
                    effectiveness = "✅ 高"
                elif abs(correlation) > 0.15:
                    effectiveness = "🟡 中"
                elif abs(correlation) > 0.05:
                    effectiveness = "🟠 低"
                else:
                    effectiveness = "❌ 极低"

                print(f"  {factor:12}   {correlation:>6.3f}    {avg_contribution:>7.3f}%  {weight:>5.2f}   {effectiveness}")

                factor_analysis.append({
                    'factor': factor,
                    'correlation': correlation,
                    'avg_contribution': avg_contribution,
                    'weight': weight,
                    'effectiveness': abs(correlation)
                })

        print()

        # 因子权重建议
        print("【因子权重调整建议】")
        print()

        factor_df = pd.DataFrame(factor_analysis).sort_values('effectiveness', ascending=False)

        print("  推荐权重调整（基于相关性）:")
        print()

        total_effectiveness = factor_df['effectiveness'].sum()
        for _, row in factor_df.iterrows():
            current_weight = row['weight']
            recommended_weight = row['effectiveness'] / total_effectiveness
            adjustment = recommended_weight - current_weight

            if abs(adjustment) > 0.05:
                arrow = "↑" if adjustment > 0 else "↓"
                print(f"    {row['factor']:12} : {current_weight:.2f} → {recommended_weight:.2f}  {arrow} ({adjustment:+.2f})")
            else:
                print(f"    {row['factor']:12} : {current_weight:.2f} → {recommended_weight:.2f}  ≈ (保持)")

        print()
else:
    print("[INFO] 需要实际表现数据才能分析因子有效性")
    print()

# ============================================================================
# 3. 股票选择分析
# ============================================================================

print("="*80)
print("【3. 股票选择分析】")
print("="*80)
print()

if 'actual_return' in signals_df.columns:
    completed = signals_df[signals_df['actual_return'].notna()].copy()

    if len(completed) > 0:
        stock_stats = completed.groupby('symbol').agg({
            'actual_return': ['count', 'mean', 'std'],
            'trade_profitable': 'sum',
            'gap_pct': 'mean',
            'prediction_error': 'mean'
        }).round(3)

        stock_stats['win_rate'] = (stock_stats[('trade_profitable', 'sum')] /
                                   stock_stats[('actual_return', 'count')] * 100)

        # 排序
        stock_stats = stock_stats.sort_values(('actual_return', 'mean'), ascending=False)

        print("各股票表现排名:")
        print()
        print("  股票   交易次数  平均收益  收益波动  胜率   平均Gap  预测误差")
        print("-" * 75)

        for symbol in stock_stats.index[:15]:  # 只显示前15名
            count = int(stock_stats.loc[symbol, ('actual_return', 'count')])
            avg_return = stock_stats.loc[symbol, ('actual_return', 'mean')]
            std_return = stock_stats.loc[symbol, ('actual_return', 'std')]
            win_rate = stock_stats.loc[symbol, ('win_rate', '')]
            avg_gap = stock_stats.loc[symbol, ('gap_pct', 'mean')]
            pred_error = stock_stats.loc[symbol, ('prediction_error', 'mean')]

            print(f"  {symbol:6}  {count:4}     {avg_return:>+6.2f}%   {std_return:>6.2f}%  {win_rate:>5.1f}%  {avg_gap:>+6.2f}%  {pred_error:>+6.2f}%")

        print()

        # 识别问题股票
        problem_stocks = stock_stats[
            (stock_stats[('actual_return', 'mean')] < 0) |
            (stock_stats[('win_rate', '')] < 40)
        ]

        if len(problem_stocks) > 0:
            print("⚠️  表现不佳的股票（考虑从池中移除）:")
            for symbol in problem_stocks.index:
                avg_return = problem_stocks.loc[symbol, ('actual_return', 'mean')]
                win_rate = problem_stocks.loc[symbol, ('win_rate', '')]
                print(f"    {symbol}: 平均收益 {avg_return:+.2f}%, 胜率 {win_rate:.1f}%")
            print()
else:
    print("[INFO] 需要实际表现数据才能分析股票选择")
    print()

# ============================================================================
# 4. Gap影响分析
# ============================================================================

print("="*80)
print("【4. Gap影响分析】")
print("="*80)
print()

if 'gap_impact' in signals_df.columns:
    completed = signals_df[signals_df['gap_impact'].notna()].copy()

    if len(completed) > 0:
        avg_gap = completed['gap_pct'].mean()
        avg_gap_impact = completed['gap_impact'].mean()
        avg_actual_return = completed['actual_return'].mean()

        gap_impact_ratio = abs(avg_gap_impact / avg_actual_return) if avg_actual_return != 0 else 0

        print(f"Gap统计:")
        print(f"  平均Gap:      {avg_gap:+.3f}%")
        print(f"  Gap影响:      {avg_gap_impact:+.3f}%")
        print(f"  影响程度:     {gap_impact_ratio*100:.1f}%")
        print()

        if gap_impact_ratio > 0.5:
            print("  🔴 Gap影响严重 (>50%) - 强烈建议重新定义Target")
        elif gap_impact_ratio > 0.35:
            print("  🟠 Gap影响较大 (35-50%) - 建议添加Gap特征并重新训练")
        elif gap_impact_ratio > 0.2:
            print("  🟡 Gap影响中等 (20-35%) - 建议添加限价单保护")
        else:
            print("  🟢 Gap影响可接受 (<20%) - 当前策略良好")

        print()
else:
    print("[INFO] 需要Gap数据才能分析影响")
    print()

# ============================================================================
# 5. 账户表现总览
# ============================================================================

print("="*80)
print("【5. 账户表现总览】")
print("="*80)
print()

if account_df is not None and len(account_df) > 0:
    initial_equity = account_df['equity'].iloc[0]
    final_equity = account_df['equity'].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100

    max_equity = account_df['equity'].max()
    min_equity = account_df['equity'].min()
    max_drawdown = (max_equity - min_equity) / max_equity * 100

    print(f"资金曲线:")
    print(f"  初始资金:     ${initial_equity:,.2f}")
    print(f"  最终资金:     ${final_equity:,.2f}")
    print(f"  总收益:       ${final_equity - initial_equity:+,.2f} ({total_return:+.2f}%)")
    print()

    print(f"风险指标:")
    print(f"  最大回撤:     {max_drawdown:.2f}%")
    print()

    # 计算Sharpe比率（如果有足够数据）
    if len(account_df) > 5:
        daily_returns = account_df['daily_pnl_pct']
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        print(f"  Sharpe比率:   {sharpe:.2f}")
        print()

# ============================================================================
# 6. 生成精调建议报告
# ============================================================================

print("="*80)
print("【6. 模型精调建议】")
print("="*80)
print()

recommendations = []

if 'actual_return' in signals_df.columns:
    completed = signals_df[signals_df['actual_return'].notna()].copy()

    if len(completed) > 0:
        # 建议1: 方向准确率
        direction_accuracy = (completed['prediction_direction_correct'] == True).sum() / len(completed) * 100

        if direction_accuracy < 55:
            recommendations.append({
                'priority': '高',
                'category': '模型架构',
                'issue': f'方向准确率较低 ({direction_accuracy:.1f}%)',
                'suggestion': '考虑增加特征数量、调整模型超参数、或尝试更复杂的模型结构'
            })
        elif direction_accuracy < 60:
            recommendations.append({
                'priority': '中',
                'category': '模型优化',
                'issue': f'方向准确率中等 ({direction_accuracy:.1f}%)',
                'suggestion': '微调模型参数、特征工程优化'
            })

        # 建议2: 预测偏差
        pred_error_mean = completed['prediction_error'].mean()

        if abs(pred_error_mean) > 0.2:
            recommendations.append({
                'priority': '高',
                'category': '预测偏差',
                'issue': f'系统性{"高估" if pred_error_mean < 0 else "低估"}收益 ({pred_error_mean:+.3f}%)',
                'suggestion': '调整模型校准、或在预测后应用偏差修正'
            })

        # 建议3: Gap影响
        if 'gap_impact' in completed.columns:
            avg_gap_impact = completed['gap_impact'].mean()
            avg_actual_return = completed['actual_return'].mean()
            gap_impact_ratio = abs(avg_gap_impact / avg_actual_return) if avg_actual_return != 0 else 0

            if gap_impact_ratio > 0.5:
                recommendations.append({
                    'priority': '高',
                    'category': 'Target定义',
                    'issue': f'Gap影响严重 ({gap_impact_ratio*100:.1f}%)',
                    'suggestion': '重新定义Target为 (next_close - next_open) / next_open，完全重新训练'
                })
            elif gap_impact_ratio > 0.35:
                recommendations.append({
                    'priority': '中',
                    'category': '特征工程',
                    'issue': f'Gap影响较大 ({gap_impact_ratio*100:.1f}%)',
                    'suggestion': '添加历史gap特征（5日均值、20日波动率），重新训练'
                })

        # 建议4: 因子权重
        factors = ['momentum', 'trend', 'volatility', 'volume', 'market', 'alpha']
        factor_correlations = {}

        for factor in factors:
            factor_col = f'factor_{factor}_pct'
            if factor_col in completed.columns:
                corr = completed[factor_col].corr(completed['actual_return'])
                factor_correlations[factor] = abs(corr)

        if len(factor_correlations) > 0:
            min_corr_factor = min(factor_correlations, key=factor_correlations.get)
            min_corr_value = factor_correlations[min_corr_factor]

            if min_corr_value < 0.05:
                recommendations.append({
                    'priority': '中',
                    'category': '因子权重',
                    'issue': f'{min_corr_factor}因子有效性极低 (相关性 {min_corr_value:.3f})',
                    'suggestion': f'降低{min_corr_factor}权重或考虑移除该因子'
                })

# 输出建议
if len(recommendations) > 0:
    priority_order = {'高': 1, '中': 2, '低': 3}
    recommendations.sort(key=lambda x: priority_order[x['priority']])

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}优先级] {rec['category']}")
        print(f"   问题: {rec['issue']}")
        print(f"   建议: {rec['suggestion']}")
        print()
else:
    print("✅ 模型表现良好，暂无重大调整建议")
    print()

# ============================================================================
# 7. 导出完整分析报告
# ============================================================================

print("="*80)
print("【7. 导出分析报告】")
print("="*80)
print()

# 生成JSON格式的完整报告
report = {
    'generated_at': datetime.now().isoformat(),
    'data_period': {
        'start_date': str(signals_df['date'].min().date()),
        'end_date': str(signals_df['date'].max().date()),
        'trading_days': int(trading_days),
        'total_trades': int(total_trades)
    },
    'recommendations': recommendations
}

if 'actual_return' in signals_df.columns:
    completed = signals_df[signals_df['actual_return'].notna()].copy()

    if len(completed) > 0:
        report['performance'] = {
            'direction_accuracy': float(direction_accuracy),
            'avg_prediction': float(completed['prediction_pct'].mean()),
            'avg_actual_return': float(completed['actual_return'].mean()),
            'win_rate': float((completed['trade_profitable'] == True).sum() / len(completed) * 100),
            'prediction_error_mean': float(completed['prediction_error'].mean()),
            'prediction_error_std': float(completed['prediction_error'].std())
        }

        if 'gap_impact' in completed.columns:
            report['gap_analysis'] = {
                'avg_gap': float(completed['gap_pct'].mean()),
                'avg_gap_impact': float(completed['gap_impact'].mean()),
                'gap_impact_ratio': float(gap_impact_ratio * 100)
            }

        # 因子分析
        if len(factor_analysis) > 0:
            report['factor_analysis'] = {
                factor['factor']: {
                    'correlation': float(factor['correlation']),
                    'avg_contribution': float(factor['avg_contribution']),
                    'current_weight': float(factor['weight'])
                }
                for factor in factor_analysis
            }

if account_df is not None and len(account_df) > 0:
    report['account_performance'] = {
        'initial_equity': float(initial_equity),
        'final_equity': float(final_equity),
        'total_return_pct': float(total_return),
        'max_drawdown_pct': float(max_drawdown)
    }

# 保存报告
report_file = logs_dir / f"full_analysis_report_{datetime.now().strftime('%Y%m%d')}.json"
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"[OK] 完整分析报告已保存: {report_file}")
print()

print("="*80)
print()
print("下一步:")
print("  1. 将 trading_logs/ 整个文件夹打包")
print("  2. 发给我进行深度分析")
print("  3. 我会基于这些数据:")
print("     - 精确调整因子权重")
print("     - 优化模型参数")
print("     - 改进特征工程")
print("     - 调整Target定义（如需要）")
print("     - 提供完整的模型精调方案")
print()
print("="*80)
