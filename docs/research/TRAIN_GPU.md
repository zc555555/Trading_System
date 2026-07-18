# GPU Training Guide for RTX 4080

## 在游戏本上使用RTX 4080进行GPU训练

### 前提条件

1. **CUDA Toolkit安装** (必需)
   - 下载地址: https://developer.nvidia.com/cuda-downloads
   - 推荐版本: CUDA 11.8 或 12.x
   - 验证安装: `nvcc --version`

2. **NVIDIA驱动** (通常游戏本已安装)
   - 验证: `nvidia-smi`
   - 应该能看到RTX 4080信息

### 步骤1: 安装GPU版本的XGBoost

在游戏本的Python环境中运行:

```bash
# 卸载CPU版本
pip uninstall xgboost

# 安装GPU版本
pip install xgboost
```

**注意**: 新版XGBoost已经统一,不需要单独的GPU版本了。只要有CUDA,就会自动使用GPU。

### 步骤2: 验证GPU可用性

```python
import xgboost as xgb
import subprocess

# 检查CUDA是否可用
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    print("\n✓ GPU is available!")
except:
    print("✗ GPU not found")
```

### 步骤3: 运行GPU训练

```bash
# 进入research目录
cd research

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 运行训练(自动检测GPU)
python train/train_xgb.py

# 强制使用GPU
python -c "from train.train_xgb import XGBoostTrainer; trainer = XGBoostTrainer(); trainer.run_full_pipeline(use_gpu=True)"
```

### 性能对比

| 配置 | 训练时间(估计) | 加速比 |
|------|--------------|--------|
| CPU (多核) | ~10-30分钟 | 1x |
| RTX 4080 (GPU) | ~1-3分钟 | **10-30x** |

RTX 4080的16GB显存完全够用!

### 监控GPU使用情况

在训练时打开另一个终端,运行:

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或Windows上
nvidia-smi -l 1
```

你应该能看到:
- GPU使用率接近100%
- 显存使用(通常几GB就够)
- 温度和功耗

### 常见问题

**Q: 显示"GPU not available"怎么办?**
A:
1. 确认安装了CUDA Toolkit
2. 确认驱动正常 (`nvidia-smi`)
3. 重启Python环境

**Q: 训练时GPU使用率很低?**
A:
- 检查是否真的在用GPU (`tree_method='gpu_hist'`应该出现在日志中)
- 数据量太小时CPU可能更快

**Q: 需要在两台电脑间同步吗?**
A:
建议流程:
1. **游戏本**: 拉取数据 → 构建数据集 → **GPU训练** → 导出ONNX
2. **当前PC**: 使用导出的ONNX模型进行推理和回测
3. 或者整个project文件夹同步(通过Git/云盘)

### 最佳实践

```python
# train/train_xgb.py 使用示例
from train.train_xgb import XGBoostTrainer

trainer = XGBoostTrainer()

# 自动检测GPU
model, metrics = trainer.run_full_pipeline()

# 强制GPU训练
model, metrics = trainer.run_full_pipeline(use_gpu=True)

# 强制CPU训练
model, metrics = trainer.run_full_pipeline(use_gpu=False)
```

### 传输模型到当前PC

训练完成后,只需复制这些文件:

```
research/artifacts/
├── model.onnx              # ONNX模型(Go使用)
├── feature_manifest.json   # 特征定义
├── golden_vectors.json     # 测试向量
└── thresholds.json         # 决策阈值
```

ONNX模型是跨平台的,在任何机器上都能用!
