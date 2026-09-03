# 🚀 快速开始指南

完整的外部数据集成系统已经开发完成！按照下面的步骤执行即可。

---

## 📋 方案选择

### 方案A：仅IV数据（1小时，快速见效）

```bash
cd C:\Trading_System\research

# 1. 获取IV数据
python data/fetch_option_iv.py

# 2. 生成IV特征
python features/process_iv_with_lstm.py

# 3. 训练模型
python train/train_with_external_features.py --iv-only
```

**预期结果**: 59.12% → 59.7-60.2%

---

### 方案B：完整流程（IV + 新闻，需要API key）

```bash
cd C:\Trading_System\research

# === 第1步：期权IV ===
python data/fetch_option_iv.py
python features/process_iv_with_lstm.py

# === 第2步：新闻情绪 ===
# 2.1 安装依赖
pip install transformers torch

# 2.2 获取API key
# 访问: https://www.alphavantage.co/support/#api-key
# 编辑 data/fetch_news_sentiment.py
# 替换第15行: ALPHA_VANTAGE_KEY = "YOUR_KEY_HERE"

# 2.3 获取新闻数据
python data/fetch_news_sentiment.py
# 选择: y (Top 5 stocks for testing)

# 2.4 生成新闻特征
python features/process_news_with_lstm.py

# === 第3步：集成训练 ===
python train/train_with_external_features.py
```

**预期结果**: 59.12% → 61-62%

---

## 📊 文件说明

### 数据获取层
- `data/fetch_option_iv.py` - 获取期权隐含波动率数据
- `data/fetch_news_sentiment.py` - 获取新闻并分析情绪

### 特征处理层
- `features/process_iv_with_lstm.py` - 生成5个IV特征
- `features/process_news_with_lstm.py` - 生成6个新闻特征

### 模型训练层
- `train/train_with_external_features.py` - 集成所有特征并训练最终模型

---

## ⚠️ 重要提示

### IV数据
- ✅ 免费，无需API key
- ✅ 立即可用
- ⏱️ 约10-15分钟

### 新闻数据
- ⚠️ 需要免费API key
- ⚠️ 免费版限制：25请求/天
- ⏱️ Top 5股票：约2分钟
- ⏱️ 全部24股票：需要24天

### 建议路线
1. **先测试IV** - 无需API，快速验证效果
2. **如果有效** - 再申请API key添加新闻数据
3. **完整集成** - 达到最佳效果

---

## 🎯 下一步

立即执行方案A：

```bash
cd C:\Trading_System\research
python data/fetch_option_iv.py
```
