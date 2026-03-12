# ESG 10-K 文本挖掘与收益方向预测

本项目基于 SEC 10-K 年报文本构建企业 ESG 相关特征，并使用 LSTM 预测企业下一期收益上涨/下跌方向。

主要流程：
1. 抓取并提取 10-K 关键章节文本（Item 1A/7/7A/8）
2. 基于 ESG 词典 + MiniLM 向量生成特征
3. 从 SEC XBRL 财务数据构建监督学习标签 `target`
4. 训练 LSTM 分类模型并输出可解释性结果（SHAP）

## 项目结构

- `extract.py`：从 `test.json` 中读取公司-URL映射，抓取 10-K 页面并抽取关键章节，输出 `esg_sections.csv`
- `delete.py`：过滤样本，保留年数不少于 8 年的公司
- `dict.py`：构建特征矩阵（ESG词典分数 + MiniLM 384维 embedding + firm/year）并输出 `feature.csv`
- `map.py`：基于净利润同比生成二分类标签，拼接得到 `feature_target.csv`
- `map1.py`：基于多财务指标加权规则生成标签，拼接得到 `feature_target1.csv`（当前训练默认使用此文件）
- `lstm.py`：训练/验证 LSTM，保存模型与配置文件
- `interpret.py`：加载已训练模型，输出 SHAP 可解释性图（`shap_summary_plot.png`）
- `feature_target1.csv`：示例特征+标签文件（已存在）
- `esg_final_lstm_model.pth` / `esg_lstm_config.pth`：已训练模型与配置（已存在）
- `test.json` / `test1.json`：公司与10-K链接映射输入

## 运行环境

建议 Python 3.10+。

### 依赖安装

```bash
pip install pandas numpy requests beautifulsoup4 tqdm scikit-learn matplotlib
pip install torch torchvision torchaudio
pip install transformers sentence-transformers shap
```

说明：
- `dict.py` 依赖 `sentence-transformers`（本地 MiniLM 模型目录）
- `lstm.py` / `interpret.py` 依赖 `torch`
- `interpret.py` 依赖 `shap`

## 数据与路径配置（重要）

当前脚本中大量使用了硬编码路径（例如 `/home/featurize/work/esg/...`）。在本地运行前，请先修改各脚本顶部常量，例如：

- `extract.py`：`INPUT_JSON`、`OUTPUT_CSV`
- `delete.py`：`INPUT_CSV`、`OUTPUT_CSV`
- `dict.py`：输入CSV路径、MiniLM本地模型路径、输出 `feature.csv` 路径
- `map.py` / `map1.py`：`INPUT_JSON`、`EXISTING_CSV`、`UPDATED_CSV`
- `lstm.py`：`DATA_PATH`
- `interpret.py`：`DATA_PATH`

建议统一改为项目内相对路径，便于复现。

## 端到端使用流程

### 1) 抽取 10-K 章节文本

```bash
python extract.py
```

输入：`test.json`（公司 -> 10-K URL 列表）  
输出：`esg_sections.csv`

### 2) 文本清洗与样本过滤（可选）

如果你有中间清洗文件（如 `preprocessed_esg_data.csv`）：

```bash
python delete.py
```

输出：`preprocessed_esg_data1.csv`

### 3) 构建特征矩阵

```bash
python dict.py
```

输出：`feature.csv`

### 4) 生成监督标签并拼接

二选一：

- 净利润同比标签：
```bash
python map.py
```
输出：`feature_target.csv`

- 多指标加权标签（推荐，与你现有训练文件一致）：
```bash
python map1.py
```
输出：`feature_target1.csv`

### 5) 训练 LSTM 模型

```bash
python lstm.py
```

输出：
- `esg_final_lstm_model.pth`
- `esg_lstm_config.pth`

### 6) 模型可解释性分析（SHAP）

```bash
python interpret.py
```

输出：`shap_summary_plot.png`

## 标签定义说明

- `map.py`：
  - 若 `target_year` 净利润 > `target_year-1` 净利润，则标签为 `1`
  - 否则为 `0`

- `map1.py`：
  - 使用四类财务指标（Cash Flow / Operating Income / Revenue / Net Income）
  - 采用权重：`0.40 / 0.30 / 0.20 / 0.10`
  - 根据加权上升比例得到二分类标签

## 模型说明（lstm.py）

- 输入：按公司分组、按年份排序后的时序特征
- 序列长度：`SEQ_LEN=3`
- 切分方式：每家公司前 `TRAIN_YEARS_NUM=5` 年训练，其后年份验证
- 输出：二分类概率（Sigmoid）

## 常见问题

1. `403 / 429`（SEC访问限制）
- 确保 `User-Agent` 合规（包含身份信息）
- 降低请求频率，保持重试机制

2. 文件行数不匹配
- `map.py`/`map1.py` 内已做最小行数截断；建议优先检查输入顺序是否一致

3. SHAP 很慢
- `interpret.py` 已随机抽样背景样本；仍可能耗时较长

4. 路径报错
- 优先将所有绝对路径改为项目相对路径

## 备注

仓库中已包含示例训练产物（`feature_target1.csv`、`.pth` 文件）。若只想快速验证推理与解释流程，可直接从 `lstm.py`/`interpret.py` 对应阶段运行。
