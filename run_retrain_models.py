"""
每周模型重新训练脚本 (Python版本)
建议运行时间：每周日 20:00
"""
import sys
import io
import os
import subprocess
from datetime import datetime
from pathlib import Path

# 设置UTF-8编码 (errors='replace': 任务计划重定向下不因特殊字符崩溃)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 子进程用与本进程相同的解释器(venv), 裸 'python' 会解析到系统 Python
PY = f'"{sys.executable}"'

def run_command(command, description):
    """运行命令并打印结果"""
    print(f"\n[*] {description}...")
    print(f"    命令: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # 打印输出
        if result.stdout:
            print(result.stdout)

        print(f"✅ {description} - 完成")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - 失败")
        print(f"错误码: {e.returncode}")
        if e.stdout:
            print("输出:", e.stdout)
        if e.stderr:
            print("错误:", e.stderr)
        return False

def main():
    print("\n" + "="*80)
    print("开始每周模型重新训练")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 确保在脚本目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"工作目录: {os.getcwd()}\n")

    # ========================================================================
    # STEP 1: Update Market Data
    # ========================================================================
    print("\n" + "="*80)
    print("[1/4] STEP 1: Update Market Data (获取最新数据)")
    print("="*80)
    print("\n说明：下载最新的市场数据，包括本周的交易数据\n")

    # 切换到research目录
    os.chdir('research')

    # 使用正确的文件路径
    data_download_cmd = f'{PY} {os.path.join("data", "fetch_ohlcv.py")}'

    if not run_command(data_download_cmd, "下载市场数据"):
        print("\n[ERROR] Data download failed!")
        os.chdir('..')
        return False

    # 流动性过滤 + 特征重建。此前重训流程抓完原始数据后直接训练,
    # 模型学的是上一次构建的旧特征 —— 必须先重建特征再训练。
    if not run_command(f'{PY} {os.path.join("data", "apply_liquidity_filter.py")}',
                       "流动性过滤"):
        print("\n[ERROR] Liquidity filter failed!")
        os.chdir('..')
        return False

    if not run_command(f'{PY} prepare_prediction_data.py', "重建特征"):
        print("\n[ERROR] Feature rebuild failed!")
        os.chdir('..')
        return False

    # 切回主目录
    os.chdir('..')

    # ========================================================================
    # STEP 2: Train Multi-Factor Models
    # ========================================================================
    print("\n" + "="*80)
    print("[2/3] STEP 2: Train Multi-Factor Models (训练多因子模型)")
    print("="*80)
    print("\n说明：训练6个因子模型（Momentum, Trend, Volatility, Volume, Market, Alpha）")
    print("每个因子使用XGBoost + LightGBM + CatBoost集成")
    print("这一步可能需要10-30分钟，请耐心等待...\n")

    train_cmd = f'{PY} {os.path.join("research", "train_multi_factor_models.py")}'

    if not run_command(train_cmd, "训练多因子集成模型"):
        print("\n[ERROR] Model training failed!")
        return False

    # ========================================================================
    # STEP 3: Validate New Models
    # ========================================================================
    print("\n" + "="*80)
    print("[3/3] STEP 3: Validate New Models (验证新模型)")
    print("="*80)
    print("\n说明：验证各因子模型是否成功保存\n")

    # 检查模型文件
    artifacts_dir = Path("research") / "artifacts"

    print("模型文件检查:")
    models = {
        "Momentum": artifacts_dir / "ensemble_momentum.pkl",
        "Trend": artifacts_dir / "ensemble_trend.pkl",
        "Volatility": artifacts_dir / "ensemble_volatility.pkl",
        "Volume": artifacts_dir / "ensemble_volume.pkl",
        "Market": artifacts_dir / "ensemble_market.pkl",
        "Alpha": artifacts_dir / "ensemble_alpha.pkl"
    }

    all_exist = True
    for name, path in models.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {exists}")
        if not exists:
            all_exist = False

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("重新训练完成!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if all_exist:
        print("✅ 新模型已保存到 research/artifacts/")
        print("   - ensemble_momentum.pkl")
        print("   - ensemble_trend.pkl")
        print("   - ensemble_volatility.pkl")
        print("   - ensemble_volume.pkl")
        print("   - ensemble_market.pkl")
        print("   - ensemble_alpha.pkl")
        print()
        print("下一步：")
        print("  1. 运行 python run_auto_trading.py 将自动使用新模型")
        print("  2. 监控Paper Trading性能，对比新旧模型表现")
    else:
        print("⚠️ 部分模型文件未找到，请检查训练过程中的错误")

    print("\n" + "="*80)

    return all_exist

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
