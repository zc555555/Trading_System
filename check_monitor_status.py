"""检查监控状态"""
import sys
import io
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_scheduled_task():
    """检查计划任务状态"""
    print("\n" + "="*80)
    print("1. 计划任务状态")
    print("="*80)

    try:
        # 使用cmd运行schtasks命令
        result = subprocess.run(
            ['cmd', '/c', 'schtasks', '/query', '/FO', 'LIST'],
            capture_output=True,
            text=True,
            encoding='gbk'  # Windows中文系统使用GBK
        )

        # 查找与交易相关的任务
        tasks = result.stdout
        found_tasks = []

        for line in tasks.split('\n'):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['alpaca', 'trading', 'monitor', 'stop']):
                found_tasks.append(line)

        if found_tasks:
            print("\n发现以下交易相关的计划任务:")
            for task in found_tasks:
                print(f"  {task.strip()}")
        else:
            print("\n❌ 未找到交易相关的计划任务")
            print("\n提示：你可能需要设置计划任务来自动运行监控")

    except Exception as e:
        print(f"\n⚠️ 无法查询计划任务: {e}")

def check_running_processes():
    """检查正在运行的Python进程"""
    print("\n" + "="*80)
    print("2. 运行中的监控进程")
    print("="*80)

    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            encoding='gbk'
        )

        python_processes = []
        for line in result.stdout.split('\n')[1:]:  # 跳过标题行
            if line.strip():
                python_processes.append(line)

        if python_processes:
            print(f"\n找到 {len(python_processes)} 个Python进程:")
            for proc in python_processes[:10]:  # 只显示前10个
                print(f"  {proc.strip()}")

            if len(python_processes) > 10:
                print(f"  ... 还有 {len(python_processes)-10} 个进程")
        else:
            print("\n❌ 没有运行中的Python进程")

    except Exception as e:
        print(f"\n⚠️ 无法查询进程: {e}")

def check_recent_logs():
    """检查最近的监控日志"""
    print("\n" + "="*80)
    print("3. 最近的监控日志")
    print("="*80)

    logs_dir = Path(__file__).parent / "trading_logs"

    if not logs_dir.exists():
        print("\n❌ 日志目录不存在")
        return

    # 检查动态交易/止损事件日志（包含旧的 stop_loss_events_*.json）
    event_logs = sorted(
        list(logs_dir.glob("dynamic_trading_events_*.json"))
        + list(logs_dir.glob("stop_loss_events_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if event_logs:
        latest_log = event_logs[0]
        file_date = latest_log.stem.split('_')[-1]

        try:
            with open(latest_log, 'r') as f:
                events = json.load(f)

            print(f"\n最新交易事件日志: {latest_log.name}")
            print(f"  日期: {file_date}")
            print(f"  事件数: {len(events)}")

            if events:
                print(f"  最近事件:")
                for event in events[-3:]:  # 显示最近3个事件
                    reason = event.get('reason') or event.get('action', 'Unknown')
                    timestamp = event.get('timestamp', 'Unknown')
                    print(f"    - {reason} at {timestamp}")
        except Exception as e:
            print(f"  ⚠️ 无法读取日志: {e}")
    else:
        print("\n❌ 没有找到交易事件日志")

    # 检查每日报告
    reports = sorted(logs_dir.glob("daily_report_*.txt"), reverse=True)

    if reports:
        latest_report = reports[0]
        file_date = latest_report.stem.split('_')[-1]
        file_time = datetime.fromtimestamp(latest_report.stat().st_mtime)

        print(f"\n最新每日报告: {latest_report.name}")
        print(f"  日期: {file_date}")
        print(f"  生成时间: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 读取报告内容
        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 显示前15行
                print(f"\n  内容预览:")
                for line in lines[:15]:
                    print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"  ⚠️ 无法读取报告: {e}")
    else:
        print("\n❌ 没有找到每日报告")

def check_account_status():
    """检查账户状态"""
    print("\n" + "="*80)
    print("4. 当前账户状态")
    print("="*80)

    try:
        from alpaca.trading.client import TradingClient
        from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY

        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()

        print(f"\n账户模式: {'PAPER TRADING' if account.account_blocked else 'LIVE'}")
        print(f"总资产: ${float(account.equity):,.2f}")
        print(f"现金: ${float(account.cash):,.2f}")
        print(f"持仓数: {len(positions)}")

        if positions:
            print(f"\n当前持仓:")
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                current_price = float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100

                print(f"  {symbol}: {qty:.0f}股 @ ${current_price:.2f} "
                      f"(盈亏: ${unrealized_pl:+.2f} / {unrealized_plpc:+.2f}%)")

    except ImportError:
        print("\n⚠️ 无法导入Alpaca库")
    except Exception as e:
        print(f"\n⚠️ 无法连接到Alpaca: {e}")

def main():
    print("="*80)
    print("监控状态检查工具")
    print("="*80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    check_scheduled_task()
    check_running_processes()
    check_recent_logs()
    check_account_status()

    print("\n" + "="*80)
    print("检查完成")
    print("="*80)

    print("\n💡 提示:")
    print("  - 如果没有找到计划任务，运行: python setup_monitor.py")
    print("  - 手动检查持仓止损/止盈: python monitor_dynamic_trading.py")
    print("  - 自动循环监控:           python monitor_dynamic_trading.py --auto")
    print("  - 查看详细账户信息:       python check_account.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
