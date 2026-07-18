# 🚀 外部数据集成指南

**目标**：通过新闻情绪和期权IV数据提升准确率 +2-3.5%

**当前准确率**：59.12%
**目标准确率**：61-62%（接近63%）

---

## 📋 实施步骤

### 阶段1：期权IV数据（1小时）⭐ 最快见效

#### 步骤1.1：获取IV数据

```bash
cd research
python data/fetch_option_iv.py
```

**输出**：`data/option_iv.parquet`

**说明**：
- 使用Yahoo Finance免费数据
- 计算历史隐含波动率
- 下载VIX（市场波动率）作为参考
- 预计耗时：10-15分钟

#### 步骤1.2：生成IV特征

```bash
python features/process_iv_with_lstm.py
```

**输出**：`data/iv_features.parquet`

**生成的5个新特征**：
1. `iv_trend` - IV趋势（上升/下降）
2. `iv_volatility` - IV波动率
3. `iv_regime` - 波动率状态（低/中/高）
4. `vix_iv_spread` - 市场vs个股波动差
5. `iv_mean_reversion` - 均值回归信号

**预期提升**：+0.5-1.0%

---

### 阶段2：新闻情绪数据（需要API，2-3天）⭐⭐ 效果最好

#### 步骤2.1：获取API Key（免费）

1. 访问：https://www.alphavantage.co/support/#api-key
2. 填写邮箱获取免费API key
3. 替换`data/fetch_news_sentiment.py`中的`ALPHA_VANTAGE_KEY`

**限制**：
- 免费版：25请求/天，500请求/月
- 24只股票需要24天（或付费升级）
- 建议：先用Top 5股票测试

#### 步骤2.2：安装依赖

```bash
# 安装transformers（FinBERT需要）
pip install transformers torch

# 首次运行会下载FinBERT模型（~500MB）
```

#### 步骤2.3：获取新闻情绪

```bash
python data/fetch_news_sentiment.py

# 选择：使用Top 5股票（y）或全部24只（n）
```

**输出**：`data/news_sentiment.parquet`

**说明**：
- 使用FinBERT分析金融新闻情绪
- 每只股票约15秒（含API限速）
- Top 5股票：约2分钟
- 全部24只：分多天运行（25/天限制）

#### 步骤2.4：生成情绪特征

```bash
python features/process_news_with_lstm.py
```

**输出**：`data/news_features.parquet`

**生成的6个新特征**：
1. `news_sentiment_trend` - 情绪趋势
2. `news_sentiment_volatility` - 情绪波动
3. `recent_negative_spike` - 突发负面新闻
4. `news_sentiment_momentum` - 情绪动量
5. `news_coverage_intensity` - 新闻热度
6. `news_sentiment_divergence` - 情绪分歧度

**预期提升**：+1.5-2.5%

---

### 阶段3：集成所有特征并重新训练（30分钟）

#### 步骤3.1：合并所有特征

```bash
python train/train_with_external_features.py
```

**功能**：
- 合并原有60特征
- 加入5个IV特征
- 加入6个新闻特征
- 总计：71个特征

#### 步骤3.2：训练最终模型

自动执行：
1. 训练XGBoost、LightGBM、CatBoost
2. 优化集成权重
3. 评估最终准确率

**预期结果**：61-62%方向准确率

---

## 🎯 快速开始（仅IV数据）

如果您想立即看到效果，先从IV数据开始：

```bash
# 1. 获取IV数据（10分钟）
python data/fetch_option_iv.py

# 2. 生成IV特征（1分钟）
python features/process_iv_with_lstm.py

# 3. 训练模型（15分钟）
python train/train_with_external_features.py --iv-only
```

**预期提升**：59.12% → 59.7-60.2%

---

## 📊 完整流程（IV + 新闻）

```bash
# === 期权IV（立即可做）===
python data/fetch_option_iv.py
python features/process_iv_with_lstm.py

# === 新闻情绪（需要API key）===
# 1. 获取API key
# 2. 编辑 data/fetch_news_sentiment.py，替换 ALPHA_VANTAGE_KEY
# 3. 安装依赖
pip install transformers torch

# 4. 获取新闻（Top 5股票，2分钟）
python data/fetch_news_sentiment.py
# 选择 y (Top 5)

# 5. 生成特征
python features/process_news_with_lstm.py

# === 集成训练 ===
python train/train_with_external_features.py
```

**预期结果**：59.12% → 61-62%

---

## ⚠️ 重要提醒

### 数据质量

**期权IV**：
- ✅ 免费，立即可用
- ✅ 数据质量高（Yahoo Finance）
- ⚠️ 使用实现波动率作为IV代理

**新闻情绪**：
- ⚠️ 需要API key（免费但有限制）
- ⚠️ 免费版每天25请求
- ✅ FinBERT是专业金融情感模型
- 建议：先测试Top 5，效果好再扩展

### 付费选项（可选）

如果效果好，想要更多数据：

1. **Alpha Vantage高级版**（$50/月）
   - 无限API请求
   - 实时新闻
   - 更多历史数据

2. **真实期权IV数据**
   - CBOE数据服务
   - OptionMetrics（学术）
   - 获取真实历史IV（而非代理）

3. **专业新闻源**
   - Bloomberg API
   - Reuters NewsScope
   - 更全面的新闻覆盖

---

## 🔍 常见问题

### Q1: 我没有API key，能用吗？

**A**: 可以先用期权IV数据（不需要API），预期+0.5-1%提升。

### Q2: FinBERT下载失败怎么办？

**A**: 脚本会自动回退到简单的关键词情感分析（效果略差但可用）。

### Q3: 24只股票要24天才能获取完新闻？

**A**: 是的（免费版限制）。建议：
- 先测试Top 5（2分钟）
- 如果效果好，再慢慢收集全部
- 或者付费升级API

### Q4: IV特征为什么不用LSTM？

**A**: 简单的统计特征（趋势、波动率）已经足够有效，更快更稳定。真正需要LSTM的是文本序列（新闻）。

---

## 📈 预期结果

| 阶段 | 准确率 | 提升 | 耗时 |
|------|--------|------|------|
| 当前（超级集成） | 59.12% | - | - |
| + IV特征 | 59.7-60.2% | +0.6-1.1% | 1小时 |
| + 新闻特征（Top 5） | 60.5-61.2% | +0.8-1.0% | +2分钟 |
| + 新闻特征（全部） | 61.0-62.0% | +0.5-0.8% | +多天 |
| **总计** | **61-62%** | **+1.9-2.9%** | **1天-2周** |

---

## 🚀 立即开始！

**推荐路线**：

```bash
# 第1步：IV数据（立即，1小时）
python data/fetch_option_iv.py
python features/process_iv_with_lstm.py

# 第2步：训练并查看效果
python train/train_with_external_features.py --iv-only

# 如果效果好（>59.7%），继续：
# 第3步：获取API key并添加新闻

# 第4步：最终训练
python train/train_with_external_features.py
```

**预期最终结果**：61-62%方向准确率！

---

**准备好开始了吗？运行第一个命令：**

```bash
python data/fetch_option_iv.py
```
