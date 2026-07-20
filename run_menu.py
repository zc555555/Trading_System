#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主菜单 - Python版本（解决批处理文件编码问题）
"""

import os
import sys
import subprocess
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[错误] 命令执行失败！")
        input("\n按回车键继续...")

    return result.returncode == 0

def show_menu():
    """显示主菜单"""
    clear_screen()
    print("\n" + "="*80)
    print(" " * 25 + "股票量化交易系统 - 主控制台")
    print("="*80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("="*80)
    print(" " * 30 + "日常操作")
    print("="*80)
    print("\n  [1] 今晚交易 (21:00运行) - 自动选股买入")
    print("  [2] 查看实时仪表板 - 监控持仓和盈亏")
    print("  [3] 查看当前持仓状态 - 快速检查\n")

    print("="*80)
    print(" " * 30 + "分析工具")
    print("="*80)
    print("\n  [4] 市场状态分析 - 查看牛市/熊市")
    print("  [5] 归因分析 - 为什么赚钱/亏钱")
    print("  [6] 数据质量检查 - 确保数据健康\n")

    print("="*80)
    print(" " * 30 + "优化工具")
    print("="*80)
    print("\n  [7] 参数优化 - 找最优止盈止损")
    print("  [8] 重新训练模型 (每周运行) - 更新策略")
    print("  [9] 回测验证 - 测试策略表现\n")

    print("="*80)
    print(" " * 30 + "设置管理")
    print("="*80)
    print("\n  [S] 设置监控系统 - 首次使用必须运行")
    print("  [H] 使用帮助 - 查看详细说明")
    print("  [Q] 退出\n")
    print("="*80 + "\n")

def main():
    """主函数"""
    # 确保在脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        show_menu()
        choice = input("请选择操作 [1-9/S/H/Q]: ").strip().upper()

        if choice == '1':
            # 自动交易
            run_auto_trading()

        elif choice == '2':
            # 仪表板
            clear_screen()
            print("\n启动仪表板...")
            print("浏览器将自动打开 http://localhost:8501")
            print("按 Ctrl+C 可以停止仪表板\n")
            input("按回车键启动...")
            subprocess.Popen(['streamlit', 'run', 'dashboard.py'])
            print("\n仪表板已在后台启动！")
            input("\n按回车键返回主菜单...")

        elif choice == '3':
            # 查看状态
            run_command('python monitor_dynamic_trading.py', '[3] 查看当前持仓状态')
            input("\n按回车键继续...")

        elif choice == '4':
            # 市场分析
            run_command('python market_regime.py', '[4] 市场状态分析')
            input("\n按回车键继续...")

        elif choice == '5':
            # 归因分析
            run_attribution_analysis()

        elif choice == '6':
            # 数据检查
            run_command('python data_quality_monitor.py', '[6] 数据质量检查')
            input("\n按回车键继续...")

        elif choice == '7':
            # 参数优化
            run_command('python parameter_optimizer.py', '[7] 参数优化')
            input("\n按回车键继续...")

        elif choice == '8':
            # 重新训练
            clear_screen()
            print("\n" + "="*80)
            print("[8] 重新训练模型")
            print("="*80)
            print("\n这将使用最新数据重新训练所有模型")
            print("建议运行频率：每周一次")
            print("预计耗时：10-30分钟\n")
            confirm = input("确认开始训练? [Y/N]: ").strip().upper()
            if confirm == 'Y':
                os.chdir('research')
                run_command('python train_multi_factor_models.py', '训练多因子模型')
                os.chdir('..')
            input("\n按回车键继续...")

        elif choice == '9':
            # 回测验证
            os.chdir('research')
            run_command('python backtest_multi_factor.py', '[9] 回测验证')
            os.chdir('..')
            input("\n按回车键继续...")

        elif choice == 'S':
            # 设置监控
            clear_screen()
            print("\n" + "="*80)
            print("[S] 设置监控系统")
            print("="*80)
            print("\n这将设置定时监控任务（必须以管理员身份运行）\n")
            print("功能：")
            print("  - 每5分钟自动检查持仓")
            print("  - 自动止盈止损")
            print("  - 21:00自动平仓\n")
            print("运行时段：14:30 - 21:05（英国时间）\n")
            confirm = input("确认设置? [Y/N]: ").strip().upper()
            if confirm == 'Y':
                run_command('python setup_smart_monitor.py', '设置监控系统')
            input("\n按回车键继续...")

        elif choice == 'H':
            # 帮助
            clear_screen()
            help_path = os.path.join('docs', 'zh', 'usage-guide.zh.txt')
            if os.path.exists(help_path):
                with open(help_path, 'r', encoding='utf-8') as f:
                    print(f.read())
            else:
                print(f"\n未找到使用指南文件: {help_path}")
            input("\n按回车键继续...")

        elif choice == 'Q':
            # 退出
            clear_screen()
            print("\n感谢使用股票量化交易系统！\n")
            break

        else:
            print("\n[错误] 无效选择，请重新输入")
            input("\n按回车键继续...")

def run_auto_trading():
    """运行自动交易流程

    完整管线只在 run_auto_trading.py 维护一份(含流动性过滤和
    legacy/staggered 策略分支), 菜单直接委托, 避免两份流程漂移。
    """
    clear_screen()
    print("\n" + "="*80)
    print("自动交易 - 完整流程")
    print("="*80)
    try:
        import config_trading
        print(f"\n当前策略: {config_trading.STRATEGY}")
    except Exception:
        pass
    print("\n这将执行完整的交易流程：")
    print("  1. 下载最新数据 + 流动性过滤")
    print("  2. 生成交易信号")
    print("  3. 按策略执行交易 (legacy: 平仓后买入 / staggered: 批次开平仓)")
    print("\n建议运行时间：每天21:00（英国时间）\n")

    confirm = input("确认开始交易? [Y/N]: ").strip().upper()
    if confirm != 'Y':
        return

    run_command('python run_auto_trading.py', '自动交易完整流程')

    input("\n按回车键返回主菜单...")

def run_attribution_analysis():
    """运行归因分析"""
    clear_screen()
    print("\n" + "="*80)
    print("[5] 归因分析")
    print("="*80)
    print("\n选择归因分析类型：")
    print("  [1] 交易归因 - 哪些股票/交易赚钱")
    print("  [2] 因子归因 - 哪些因子贡献大")
    print("  [3] 两者都运行\n")

    attr_choice = input("选择 [1-3]: ").strip()

    if attr_choice == '1':
        run_command('python attribution_analysis.py', '交易归因分析')
    elif attr_choice == '2':
        run_command('python factor_attribution.py', '因子归因分析')
    elif attr_choice == '3':
        run_command('python attribution_analysis.py', '交易归因分析')
        print("\n")
        run_command('python factor_attribution.py', '因子归因分析')
    else:
        print("\n无效选择")

    input("\n按回车键继续...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序。")
        sys.exit(0)
