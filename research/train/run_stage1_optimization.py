"""
阶段1优化：一键运行所有动态股票池策略

运行三种策略并对比结果：
1. 基于置信度的动态Top股票
2. 基于滚动窗口的动态选股
3. 核心-卫星混合策略

输出：
- 各策略最佳配置
- 性能对比
- 推荐方案

预期：59.24% -> 62%+
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_script(script_name):
    """运行Python脚本并捕获输出"""
    print(f"\n{'=' * 80}")
    print(f"运行: {script_name}")
    print(f"{'=' * 80}\n")

    script_path = Path(__file__).parent / script_name
    python_exe = r"C:\Users\13785\AppData\Local\Programs\Python\Python312\python.exe"

    try:
        result = subprocess.run(
            [python_exe, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        print(result.stdout)
        if result.stderr:
            print(f"[警告] {result.stderr}")

        return result.returncode == 0
    except Exception as e:
        print(f"[错误] 运行失败: {e}")
        return False


def load_results():
    """加载所有策略的结果"""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    results = {}

    # 1. 动态Top股票
    path1 = artifacts_dir / "dynamic_top_stocks_results.json"
    if path1.exists():
        with open(path1, 'r', encoding='utf-8') as f:
            results['dynamic_top'] = json.load(f)

    # 2. 滚动窗口
    path2 = artifacts_dir / "rolling_window_selection_results.json"
    if path2.exists():
        with open(path2, 'r', encoding='utf-8') as f:
            results['rolling_window'] = json.load(f)

    # 3. 核心-卫星
    path3 = artifacts_dir / "core_satellite_results.json"
    if path3.exists():
        with open(path3, 'r', encoding='utf-8') as f:
            results['core_satellite'] = json.load(f)

    return results


def compare_results(results):
    """对比所有策略的结果"""
    print("\n" + "=" * 80)
    print("阶段1优化结果对比")
    print("=" * 80)

    baseline = None
    comparisons = []

    for strategy_name, data in results.items():
        if 'baseline_accuracy' in data:
            baseline = data['baseline_accuracy']

        if 'best_config' in data:
            best = data['best_config']
            comparisons.append({
                'strategy': strategy_name,
                'accuracy': best['overall_accuracy'],
                'description': best.get('description', 'N/A'),
                'config': best.get('config', {})
            })

    if baseline:
        print(f"\n基线准确率（所有股票，无筛选）: {baseline*100:.2f}%")

    print(f"\n{'策略':<20} {'准确率':<12} {'提升':<10} {'配置':<50}")
    print("-" * 95)

    comparisons.sort(key=lambda x: x['accuracy'], reverse=True)

    for comp in comparisons:
        strategy_names = {
            'dynamic_top': '动态Top股票',
            'rolling_window': '滚动窗口',
            'core_satellite': '核心-卫星'
        }

        name = strategy_names.get(comp['strategy'], comp['strategy'])
        acc = comp['accuracy']
        improvement = (acc - baseline) * 100 if baseline else 0

        # 配置简要说明
        config = comp['config']
        if 'n_top_stocks' in config:
            config_str = f"Top {config['n_top_stocks']}, 置信度 {config.get('min_confidence', 0):.0%}"
        elif 'window_days' in config:
            config_str = f"{config['window_days']}天窗口, Top {config['n_top_stocks']}, {config['rebalance_freq']}"
        elif 'n_core' in config:
            config_str = f"{config['n_core']}核心 + {config['n_satellite']}卫星, {config['core_weight']:.0%}/{1-config['core_weight']:.0%}"
        else:
            config_str = "N/A"

        print(f"{name:<20} {acc*100:>6.2f}%     {improvement:>+5.2f}pp   {config_str:<50}")

    # 推荐
    print(f"\n{'=' * 80}")
    print("推荐方案")
    print(f"{'=' * 80}")

    best_overall = comparisons[0]
    best_acc = best_overall['accuracy']

    print(f"\n最佳策略: {strategy_names.get(best_overall['strategy'], best_overall['strategy'])}")
    print(f"准确率: {best_acc*100:.2f}%")
    print(f"提升: {(best_acc - baseline)*100:+.2f} pp")

    if best_acc >= 0.62:
        print(f"\n[优秀] 达到62%目标！")
        print("建议：直接进入阶段2（个股定制模型）")
    elif best_acc >= 0.61:
        print(f"\n[良好] 接近62%目标")
        print("建议：")
        print("  1. 微调当前最佳策略的参数")
        print("  2. 结合高置信度过滤进一步提升")
        print("  3. 或直接进入阶段2")
    elif best_acc >= 0.60:
        print(f"\n[及格] 突破60%")
        print("建议：")
        print("  1. 尝试组合多个策略")
        print("  2. 结合样本质量筛选")
        print("  3. 进入阶段2进一步优化")
    else:
        print(f"\n[一般] 未达60%")
        print("建议：")
        print("  1. 检查数据质量")
        print("  2. 重新评估特征重要性")
        print("  3. 考虑其他优化方向")

    # 保存综合报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline_accuracy': float(baseline) if baseline else None,
        'strategies': comparisons,
        'best_strategy': {
            'name': best_overall['strategy'],
            'accuracy': float(best_overall['accuracy']),
            'improvement': float((best_overall['accuracy'] - baseline) * 100) if baseline else None,
            'config': best_overall['config']
        }
    }

    report_path = Path(__file__).parent.parent / "artifacts" / "stage1_optimization_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n综合报告已保存: {report_path}")


def main():
    """主函数：运行所有阶段1优化策略"""

    print("\n" + "=" * 80)
    print("阶段1优化 - 动态股票池策略测试")
    print("=" * 80)
    print("\n将依次运行以下策略：")
    print("  1. 基于置信度的动态Top股票")
    print("  2. 基于滚动窗口的动态选股")
    print("  3. 核心-卫星混合策略")
    print("\n预计运行时间：5-10分钟")

    input("\n按Enter键开始...")

    # 运行策略1
    success1 = run_script("train_dynamic_top_stocks.py")

    # 运行策略2
    success2 = run_script("train_rolling_window_selection.py")

    # 运行策略3
    success3 = run_script("train_core_satellite_strategy.py")

    # 汇总结果
    if success1 or success2 or success3:
        print("\n" + "=" * 80)
        print("加载并对比结果...")
        print("=" * 80)

        results = load_results()

        if results:
            compare_results(results)
        else:
            print("[错误] 未找到结果文件")
    else:
        print("\n[错误] 所有策略均运行失败")

    print("\n" + "=" * 80)
    print("阶段1优化完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
