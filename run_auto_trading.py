#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动交易 - Python版本（解决批处理文件编码问题）

支持两种运行方式:
  交互式  -- 双击 .bat / 终端运行, 出错和结束时等待按回车
  无人值守 -- Windows 任务计划调用(stdin 非终端), 跳过所有交互,
             以退出码报告成败, 输出重定向到日志安全(强制 UTF-8)
"""

import io
import os
import sys
import subprocess
from datetime import datetime

# 任务计划下 stdout 是管道, Windows 默认 cp1252 编码会在中文输出上崩溃
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

INTERACTIVE = sys.stdin is not None and sys.stdin.isatty()

# 子进程必须用与本进程相同的解释器(venv), 裸 'python' 会解析到系统
# Python, 包版本和 venv 不一致。
PY = f'"{sys.executable}"'


def pause_exit(code=1):
    """交互模式下等待按回车; 无人值守模式下直接以退出码结束。"""
    if INTERACTIVE:
        input("\n按回车键退出...")
    sys.exit(code)


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[错误] {description} 失败！")
        return False

    print(f"\n[完成] {description} 成功！")
    return True

def main():
    """自动交易主流程"""
    # 确保在脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\n" + "="*80)
    print(" " * 25 + "自动交易工作流程")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 检查配置文件
    if not os.path.exists('config_alpaca.py'):
        print("[错误] 未找到 config_alpaca.py！")
        print("请先创建配置文件并填入API密钥。")
        pause_exit(1)

    # Step 1: 更新数据
    print("\n" + "="*80)
    print("[1/5] 步骤1: 更新市场数据和生成信号")
    print("="*80)

    os.chdir('research')

    print("\n[1/4] 下载最新股票数据...")
    if not run_command(f'{PY} {os.path.join("data", "fetch_ohlcv.py")}', '下载OHLCV数据'):
        os.chdir('..')
        pause_exit(1)

    print("\n[2/4] 应用流动性过滤...")
    if not run_command(f'{PY} {os.path.join("data", "apply_liquidity_filter.py")}', '流动性过滤'):
        os.chdir('..')
        pause_exit(1)

    print("\n[3/4] 计算技术指标...")
    if not run_command(f'{PY} prepare_prediction_data.py', '计算特征'):
        os.chdir('..')
        pause_exit(1)

    print("\n[4/4] 生成交易信号...")
    if not run_command(f'{PY} get_daily_signals_multi_factor.py', '生成交易信号'):
        os.chdir('..')
        pause_exit(1)

    os.chdir('..')

    # Step 2: 测试连接
    print("\n" + "="*80)
    print("[2/5] 步骤2: 检查Alpaca连接")
    print("="*80)
    if not run_command(f'{PY} test_alpaca_connection.py', '测试Alpaca连接'):
        pause_exit(1)

    # Step 3: 记录数据
    print("\n" + "="*80)
    print("[3/5] 步骤3: 记录完整交易数据")
    print("="*80)
    run_command(f'{PY} log_comprehensive_trading.py', '记录交易数据')

    # 执行路径由 config_trading.STRATEGY 决定:
    #   staggered — 多日错峰持仓, 由 run_staggered_trading.py 管理开平仓,
    #               绝不能在这里全量平仓(会摧毁未到期的 tranche 并使注册表失去同步)
    #   legacy    — 单日持仓, 先全量平仓再按信号买入
    import config_trading

    if config_trading.STRATEGY == "staggered":
        print("\n" + "="*80)
        print("[4/5] 步骤4-5: STAGGERED 策略 — 平掉到期批次并开新批次")
        print("="*80)
        if not run_command(f'{PY} run_staggered_trading.py', '执行错峰交易'):
            pause_exit(1)
    else:
        # Step 4: 平仓 (仅 legacy 单日策略)
        print("\n" + "="*80)
        print("[4/5] 步骤4: 平仓旧持仓")
        print("="*80)

        try:
            from monitor_dynamic_trading import DynamicTradingMonitor
            m = DynamicTradingMonitor()
            positions = m.client.get_all_positions()
            print(f"\n找到 {len(positions)} 个持仓")

            if positions:
                for p in positions:
                    print(f"平仓: {p.symbol}")
                    m.client.close_position(p.symbol)
                print("\n所有持仓已平仓")
            else:
                print("\n没有需要平仓的持仓")
        except Exception as e:
            print(f"\n平仓时出错: {e}")

        # Step 5: 执行交易
        print("\n" + "="*80)
        print("[5/5] 步骤5: 执行新交易")
        print("="*80)
        if not run_command(f'{PY} alpaca_trader.py', '执行交易'):
            pause_exit(1)

    # 生成报告
    print("\n生成每日报告...")
    try:
        from alpaca_trader import AlpacaAutoTrader
        t = AlpacaAutoTrader()
        t.generate_daily_report()
    except Exception as e:
        print(f"生成报告时出错: {e}")

    # 监控
    print("\n监控持仓表现...")
    run_command(f'{PY} monitor_paper_trading.py', '监控表现')

    # 完成
    print("\n" + "="*80)
    print(" " * 30 + "工作流程完成")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("查看结果:")
    print("  - trading_logs/         (交易日志)")
    print("  - 运行 dashboard.py     (查看仪表板)\n")

    print("下一步:")
    print("  - 明天 21:00: 持仓将自动平仓\n")

    if INTERACTIVE:
        input("按回车键退出...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序。")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        pause_exit(1)
