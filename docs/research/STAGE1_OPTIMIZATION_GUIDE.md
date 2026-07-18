# 阶段1优化指南：动态股票池策略

## 📋 目标

从 **59.24%** 提升到 **62%+** 的准确率

## 🎯 优化策略

本阶段实现了**3种动态股票池策略**，你可以选择其中一种或组合使用：

### 策略1：基于置信度的动态Top股票 ⭐⭐⭐

**原理：**
- 每天从所有股票中选择模型最有把握的Top N只
- 基于预测收益率的绝对值作为置信度

**优点：**
- 每天都交易最有信心的股票
- 灵活适应市场变化
- 实现简单

**缺点：**
- 换手率可能较高
- 可能过度拟合短期波动

**脚本：** `train_dynamic_top_stocks.py`

---

### 策略2：基于滚动窗口的动态选股 ⭐⭐⭐⭐

**原理：**
- 基于最近N天的实际准确率选择股票
- 每周/每月调整一次投资组合
- 更稳健，基于历史表现

**优点：**
- 更稳健（基于实际表现，不是单次预测）
- 换手率适中
- 避免追高杀低

**缺点：**
- 可能滞后于市场变化
- 需要一定的历史数据积累

**脚本：** `train_rolling_window_selection.py`

---

### 策略3：核心-卫星混合策略 ⭐⭐⭐⭐⭐（推荐）

**原理：**
- **核心池（70%资金）**：固定5只历史表现最稳定的股票
- **卫星池（30%资金）**：动态选择2-3只高置信度股票

**优点：**
- 平衡稳定性和灵活性
- 核心池提供稳定收益
- 卫星池捕捉短期机会
- 风险分散

**缺点：**
- 稍微复杂一些
- 需要同时管理两个股票池

**脚本：** `train_core_satellite_strategy.py`

---

## 🚀 快速开始

### 方法1：一键运行所有策略（推荐）

```bash
cd research/train
C:\Users\13785\AppData\Local\Programs\Python\Python312\python.exe run_stage1_optimization.py
```

这将：
1. 依次运行3种策略
2. 测试多种配置
3. 自动对比结果
4. 推荐最佳方案

**预计时间：** 5-10分钟

---

### 方法2：单独运行某个策略

#### 策略1：动态Top股票
```bash
cd research/train
C:\Users\13785\AppData\Local\Programs\Python\Python312\python.exe train_dynamic_top_stocks.py
```

#### 策略2：滚动窗口
```bash
cd research/train
C:\Users\13785\AppData\Local\Programs\Python\Python312\python.exe train_rolling_window_selection.py
```

#### 策略3：核心-卫星
```bash
cd research/train
C:\Users\13785\AppData\Local\Programs\Python\Python312\python.exe train_core_satellite_strategy.py
```

---

## 📊 输出结果

### 运行后会生成以下文件：

```
research/artifacts/
├── dynamic_top_stocks_results.json          # 策略1结果
├── rolling_window_selection_results.json    # 策略2结果
├── core_satellite_results.json              # 策略3结果
└── stage1_optimization_report.json          # 综合对比报告 ⭐
```

### 结果内容：

每个结果文件包含：
- **准确率**：各配置的方向准确率
- **覆盖率**：交易样本占总样本的比例
- **每日交易数**：平均每天交易多少只股票
- **各股票表现**：每只股票的准确率统计
- **最佳配置**：自动选出的最优参数

---

## 📈 预期结果

### 保守预期（基于历史数据分析）：

| 策略 | 预期准确率 | 提升幅度 |
|------|-----------|---------|
| 基线（无优化） | 59.24% | - |
| 动态Top股票 | 60-61% | +0.8-1.8 pp |
| 滚动窗口 | 61-62% | +1.8-2.8 pp |
| 核心-卫星 | 61-62% | +1.8-2.8 pp |

### 最佳情况：

如果结合高置信度过滤（只交易高把握的预测）：
- 准确率：62-64%
- 但覆盖率会下降到40-60%

---

## 🔧 参数调优指南

### 策略1：动态Top股票

**可调参数：**
```python
n_top_stocks = 7        # 每天选择股票数量（5-10）
min_confidence = 0.60   # 置信度阈值（0.0-0.8）
```

**建议配置：**
- **激进型**：`n_top_stocks=5, min_confidence=0.70` → 高准确率，低覆盖率
- **平衡型**：`n_top_stocks=7, min_confidence=0.60` → 适中
- **保守型**：`n_top_stocks=10, min_confidence=0.50` → 高覆盖率

---

### 策略2：滚动窗口

**可调参数：**
```python
window_days = 30        # 滚动窗口天数（20-45）
n_top_stocks = 7        # 选择股票数量（5-10）
rebalance_freq = 'weekly'  # 调仓频率（daily/weekly/monthly）
```

**建议配置：**
- **灵活型**：`window_days=20, rebalance_freq='weekly'`
- **标准型**：`window_days=30, rebalance_freq='weekly'` ⭐
- **稳定型**：`window_days=45, rebalance_freq='monthly'`

---

### 策略3：核心-卫星

**可调参数：**
```python
n_core = 5              # 核心股票数量（3-7）
n_satellite = 2         # 卫星股票数量（2-3）
core_weight = 0.7       # 核心权重（0.6-0.8）
satellite_method = 'confidence'  # 卫星选择方法
rebalance_satellite = 'weekly'   # 卫星调仓频率
```

**建议配置：**
- **稳健型**：`n_core=7, n_satellite=2, core_weight=0.8` → 重核心
- **平衡型**：`n_core=5, n_satellite=2, core_weight=0.7` ⭐
- **进取型**：`n_core=5, n_satellite=3, core_weight=0.6` → 重卫星

---

## 📝 使用示例

### 示例1：快速测试所有策略

```bash
# 运行一键测试脚本
python run_stage1_optimization.py

# 查看综合报告
cat ../artifacts/stage1_optimization_report.json
```

### 示例2：深入分析某个策略

```bash
# 运行核心-卫星策略
python train_core_satellite_strategy.py

# 查看详细结果
python -c "import json; r = json.load(open('../artifacts/core_satellite_results.json')); print(json.dumps(r, indent=2, ensure_ascii=False))"
```

---

## 🎓 理解结果

### 关键指标说明：

1. **overall_accuracy（整体准确率）**
   - 所有选中交易的方向准确率
   - **主要优化目标**

2. **coverage（覆盖率）**
   - 实际交易的样本占总样本的比例
   - 覆盖率 × 准确率 = 有效收益贡献

3. **trades_per_day（每日交易数）**
   - 平均每天交易的股票数量
   - 影响交易成本

4. **improvement（提升幅度）**
   - 相比基线的提升（百分点）
   - **关键评估指标**

### 如何选择最佳策略？

**决策树：**

```
如果你重视稳定性 → 选择"核心-卫星"策略
  ├─ 风险厌恶型 → 增加核心权重（80%）
  └─ 风险中性型 → 标准配置（70%）

如果你重视灵活性 → 选择"滚动窗口"策略
  ├─ 每周调仓
  └─ 30天窗口

如果你想最大化收益 → 选择"动态Top股票"策略
  ├─ 结合高置信度过滤
  └─ 可能牺牲一些覆盖率
```

---

## ✅ 成功标准

### 阶段1完成的标准：

- [ ] 至少一种策略达到 **60%+** 准确率
- [ ] 最佳策略达到 **61-62%** 准确率
- [ ] 理解各策略的优缺点
- [ ] 选定进入阶段2的策略

### 如果达到62%+：

🎉 **恭喜！阶段1优化成功！**

可以直接进入：
- **阶段2**：个股定制模型（预期：+1 pp → 63%）
- 或者开始回测验证

### 如果只达到60-61%：

继续尝试：
- 调整参数配置
- 组合多个策略
- 结合样本质量筛选（阶段2的一部分）

---

## 🔍 故障排除

### 问题1：所有策略准确率都没提升

**可能原因：**
- 数据问题：检查`stocks_with_time_windows.parquet`是否正确
- 模型问题：确认`ensemble_time_windows.pkl`存在

**解决方法：**
```bash
# 重新生成数据
python ../features/add_time_windows.py

# 重新训练模型
python train_with_time_windows.py
```

---

### 问题2：运行出错

**常见错误：**
- `FileNotFoundError`: 检查文件路径
- `ModuleNotFoundError`: 安装依赖 `pip install -r ../requirements.txt`
- `KeyError`: 数据列名不匹配，检查特征列表

---

### 问题3：运行时间过长

**优化方法：**
- 减少测试配置数量
- 使用采样数据（修改脚本中的`df.sample(frac=0.5)`）
- 单独运行某个策略而不是全部

---

## 📚 下一步

### 如果阶段1成功（≥62%）：

**进入阶段2：中期优化**

1. **为Top 5股票训练专属模型**
   ```bash
   python train_per_stock_models.py  # 即将创建
   ```

2. **样本质量筛选**
   ```bash
   python filter_training_samples.py  # 即将创建
   ```

预期：62% → 63%+

---

### 如果阶段1部分成功（60-61%）：

**继续微调阶段1：**

1. 调整参数
2. 尝试组合策略
3. 添加更多筛选条件

---

### 如果阶段1不成功（<60%）：

**重新评估：**

1. 检查数据质量
2. 重新分析特征重要性
3. 考虑其他优化方向（深度学习、付费数据）

---

## 💡 小贴士

1. **先运行一键脚本** (`run_stage1_optimization.py`)，快速了解整体效果

2. **记录每次实验的配置和结果**，方便后续对比

3. **不要过度优化测试集**，避免过拟合

4. **关注稳定性，不只是准确率**，波动大的策略风险高

5. **实盘前一定要回测**，确保策略在不同市场环境下都有效

---

## 📞 需要帮助？

- 查看 `research/artifacts/` 下的结果文件
- 检查脚本输出的详细日志
- 阅读代码注释了解实现细节

---

**祝你优化成功！🚀**
