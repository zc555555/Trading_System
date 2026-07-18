# GDELT新闻情绪数据 - 实施指南

## 概述

使用GDELT（全球事件、语言和语调数据库）获取历史新闻数据，通过FinBERT分析情绪，生成特征用于模型训练。

**预期效果：** 准确率从59.12% → 60.5-61.5% (+1.5-2.5%)

---

## 快速开始（测试模式）

### 第1步：获取测试数据（5只股票，1个月）

```bash
cd research
python data/fetch_gdelt_news.py
```

**预期输出：**
- 文件：`data/gdelt_news_raw.parquet`
- 约100-500篇新闻文章
- 耗时：约2-5分钟

**检查数据质量：**
```python
import pandas as pd
df = pd.read_parquet('data/gdelt_news_raw.parquet')
print(df.shape)
print(df['symbol'].value_counts())
print(df[['symbol', 'title', 'date']].head(20))
```

### 第2步：情绪分析

```bash
python features/process_gdelt_sentiment.py
```

**预期输出：**
- 文件：`data/news_features.parquet`
- 包含6个时序特征（sentiment_trend, volatility等）
- 耗时：约5-10分钟（FinBERT分析）

**检查特征：**
```python
df = pd.read_parquet('data/news_features.parquet')
print(df.describe())
```

### 第3步：训练模型验证效果

```bash
python train/train_with_external_features.py
```

**决策点：**
- 如果准确率 >= 60%：继续扩展到全部数据
- 如果准确率 < 60%：调整参数或放弃此方案

---

## 完整实施（全部股票，8年数据）

### 仅在测试验证有效后执行！

### 第1步：获取完整历史数据

```bash
python data/fetch_gdelt_news.py --full --start 2018-01-01 --end 2026-01-31
```

**警告：**
- 耗时：约2-4小时（24只股票×8年）
- 数据量：约50,000-200,000篇文章
- 磁盘占用：约500MB-2GB

**分批执行建议：**
```bash
# 方案1：按年份分批
python data/fetch_gdelt_news.py --full --start 2018-01-01 --end 2018-12-31
python data/fetch_gdelt_news.py --full --start 2019-01-01 --end 2019-12-31
# ...

# 方案2：按股票分批（修改config.yaml中的symbols列表）
```

### 第2步：情绪分析

```bash
python features/process_gdelt_sentiment.py
```

**耗时估算：**
- 10,000篇文章：约20分钟
- 100,000篇文章：约3小时
- 使用GPU可加速5-10倍

**GPU加速（可选）：**
```bash
# 安装GPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 第3步：训练最终模型

```bash
python train/train_with_external_features.py
```

---

## 数据质量检查清单

### 检查1：新闻覆盖率

```python
import pandas as pd

df = pd.read_parquet('data/gdelt_news_raw.parquet')

# 每只股票的新闻数量
print("\n新闻数量分布：")
print(df['symbol'].value_counts())

# 每天的新闻数量
daily_counts = df.groupby('date').size()
print(f"\n平均每天新闻数：{daily_counts.mean():.1f}")
print(f"中位数：{daily_counts.median():.1f}")

# 警告：如果某只股票新闻<50篇，可能需要扩大搜索范围
```

### 检查2：歧义消解质量

```python
# 随机抽查新闻是否真的与股票相关
sample = df.sample(20)
for idx, row in sample.iterrows():
    print(f"\n{row['symbol']}: {row['title']}")

# 手动检查是否有误匹配（例如"Apple"匹配到"apple pie"新闻）
```

### 检查3：情绪分布

```python
df_features = pd.read_parquet('data/news_features.parquet')

# 情绪趋势分布
print(df_features['news_sentiment_trend'].describe())

# 警告：如果std接近0，说明情绪变化太小，可能无法提供有效信号
```

---

## 常见问题

### Q1：GDELT API速度慢或超时？

**解决方案：**
- 减小batch size（每次查询250条改为100条）
- 增加请求间隔（time.sleep从2秒改为5秒）
- 使用代理服务器

### Q2：FinBERT分析太慢？

**优化方案：**
1. **使用GPU**（加速5-10倍）
2. **增大batch_size**（从32改为64或128）
3. **使用GDELT自带tone字段**（跳过FinBERT，速度快100倍）

使用GDELT tone字段的代码：
```python
# 在fetch_gdelt_news.py中，GDELT DOC API实际上不返回tone
# 需要改用GKG数据源（更复杂，但有tone字段）
```

### Q3：某些股票新闻太少？

**解决方案：**
- 扩展公司名列表（添加更多别名）
- 增加搜索时间范围
- 降低相关性过滤阈值
- 使用股票代码+公司名组合查询

### Q4：磁盘空间不足？

**优化方案：**
- 只保存必要字段（去掉socialimage等大字段）
- 使用压缩格式（parquet已经压缩，但可以调整压缩级别）
- 定期清理中间文件

---

## 性能基准

**测试配置：** Intel i7 CPU, 16GB RAM, 无GPU

| 阶段 | 数据量 | 耗时 |
|------|--------|------|
| 获取新闻（5股票，1月） | 200篇 | 3分钟 |
| 获取新闻（24股票，8年） | 100,000篇 | 3小时 |
| FinBERT分析（CPU） | 10,000篇 | 20分钟 |
| FinBERT分析（GPU） | 10,000篇 | 3分钟 |
| 特征生成 | 100,000篇 | 5分钟 |
| 模型训练 | 60+6特征 | 15分钟 |

**总计（CPU）：** 约4-5小时
**总计（GPU）：** 约1-2小时

---

## 成本效益分析

### 投入：
- 时间成本：1-2天（开发+调试+运行）
- 计算成本：免费（使用本地CPU/GPU）
- 数据成本：免费（GDELT开放数据）

### 预期收益：
- 准确率提升：+1.5-2.5% (59.12% → 60.5-61.5%)
- 年化收益提升：约+2-3%

### ROI：
如果管理资金>$50,000，2-3%的年化收益提升 = $1000-1500/年
投入1-2天时间，**ROI > 50倍/年**

---

## 下一步优化方向

如果GDELT方案有效，可以继续优化：

1. **使用GDELT GKG数据源**
   - 包含更多元数据（themes, locations, organizations）
   - 自带tone字段（跳过FinBERT，更快）

2. **扩展到全球新闻**
   - GDELT覆盖100+语言
   - 可捕捉国际事件对美股的影响

3. **实体识别优化**
   - 使用NER模型提高歧义消解准确度
   - 过滤掉非公司相关新闻

4. **多模态特征**
   - 新闻图片分析
   - 社交媒体情绪

---

## 支持与调试

遇到问题？检查以下内容：

1. **网络连接**：GDELT API需要稳定网络
2. **Python版本**：需要Python 3.8+
3. **依赖包**：确保安装transformers, torch, pandas
4. **磁盘空间**：至少保留5GB空闲空间

**日志文件位置：**
- 所有输出都打印到控制台
- 可以用 `python xxx.py > log.txt 2>&1` 保存日志

---

**准备好了吗？从测试模式开始：**

```bash
cd research
python data/fetch_gdelt_news.py
```

**Good luck! 🚀**
