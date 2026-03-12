#pip install "torch>=2.4" "torchvision>=0.19" "torchaudio>=2.4" --index-url https://download.pytorch.org/whl/cu121
#pip install "transformers>=4.44.0" "sentence-transformers>=3.0.0"
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# 加载预处理后的文本数据
df = pd.read_csv(r"/home/featurize/work/esg/preprocessed_esg_data1.csv", encoding="utf-8")

# 过滤空文本（双重保险）
df = df[df["processed_text"].str.len() > 10].reset_index(drop=True)
print(f"有效文本数据量：{df.shape[0]} 行")
# Hoffmann ESG词典（2019）核心词汇表
hoffmann_esg_dict = {
    # 环境（Environmental）维度
    "environment": [
        "carbon", "emission", "pollution", "sustainability", "renewable", "energy", 
        "climate", "greenhouse", "waste", "recycle", "conservation", "biodiversity",
        "ecology", "environmental", "fossil", "fuel", "water", "air", "toxic", 
        "hazardous", "compliance", "regulation", "climatechange", "globalwarming",
        "deforestation", "ecosystem", "footprint", "renewables", "cleanenergy"
    ],
    # 社会（Social）维度
    "social": [
        "labor", "employee", "diversity", "community", "safety", "humanright", 
        "customer", "stakeholder", "welfare", "health", "laborright", "fairwage",
        "equality", "inclusion", "childlabor", "forcedlabor", "workingcondition",
        "training", "education", "philanthropy", "charity", "ethical", "responsibility",
        "supplychain", "laborunion", "discrimination", "wellbeing", "worklifebalance"
    ],
    # 治理（Governance）维度
    "governance": [
        "board", "audit", "compliance", "transparency", "ethics", "integrity", 
        "accountability", "oversight", "executive", "compensation", "shareholder",
        "director", "independence", "corruption", "bribery", "fraud", "governance",
        "policy", "procedure", "internalcontrol", "riskmanagement", "disclosure",
        "nomination", "remuneration", "ethicscommittee", "auditcommittee"
    ]
}

# 统一词典词汇为小写（匹配预处理后的文本）
for category in hoffmann_esg_dict:
    hoffmann_esg_dict[category] = [word.lower() for word in hoffmann_esg_dict[category]]

def calculate_hoffmann_esg_scores(text, esg_dict):
    """
    计算Hoffmann ESG词典得分：
    - 得分逻辑：维度关键词出现次数 / 文本总词数（避免文本长度影响）
    - 返回：总ESG得分、E得分、S得分、G得分
    """
    if pd.isna(text) or text.strip() == "":
        return {"esg_total": 0.0, "e_score": 0.0, "s_score": 0.0, "g_score": 0.0}
    
    # 分词（预处理后的文本已按空格分隔）
    tokens = text.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {"esg_total": 0.0, "e_score": 0.0, "s_score": 0.0, "g_score": 0.0}
    
    # 统计各维度关键词出现次数
    e_count = sum(1 for token in tokens if token in esg_dict["environment"])
    s_count = sum(1 for token in tokens if token in esg_dict["social"])
    g_count = sum(1 for token in tokens if token in esg_dict["governance"])
    total_esg_count = e_count + s_count + g_count
    
    # 计算得分（占比）
    return {
        "esg_total": round(total_esg_count / total_tokens, 6),
        "e_score": round(e_count / total_tokens, 6),
        "s_score": round(s_count / total_tokens, 6),
        "g_score": round(g_count / total_tokens, 6)
    }

# 应用得分计算到所有文本
esg_scores = df["processed_text"].apply(lambda x: calculate_hoffmann_esg_scores(x, hoffmann_esg_dict))
esg_scores_df = pd.DataFrame(esg_scores.tolist())

# 合并ESG得分到原数据框
df = pd.concat([df, esg_scores_df], axis=1)

# 查看得分示例
print("\nESG词典得分示例：")
print(df[["firm", "year", "esg_total", "e_score", "s_score", "g_score"]].head())

def get_minilm_embeddings(texts, batch_size=16, max_length=512):
    """
    生成MiniLM向量嵌入（本地加载模型，简洁版）
    - 输入：文本列表
    - 输出：维度为 (len(texts), 384) 的向量矩阵
    """
    # ===================== 1. 配置本地模型路径 =====================
    model_path = r"/home/featurize/work/esg/sentence"
    
    # ===================== 2. 加载本地模型（自动包含tokenizer） =====================
    # SentenceTransformer会自动加载本地的tokenizer和模型，无需手动获取tokenizer
    model = SentenceTransformer(
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",  # 自动分配设备
    )
    
    # 设置文本最大长度（适配模型）
    model.max_seq_length = max_length
    
    # ===================== 3. 批量生成向量（SentenceTransformer内置封装） =====================
    # 无需手动写tokenizer/前向传播，一行代码生成向量
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,  # 显示进度条（可选，方便看进度）
        normalize_embeddings=False,  # 不标准化向量（保持原始值）
        convert_to_numpy=True  # 直接返回NumPy数组，无需手动转换
    )
    
    return np.array(embeddings)

# 生成所有预处理文本的MiniLM嵌入
print("\n开始生成MiniLM向量嵌入...")
text_embeddings = get_minilm_embeddings(df["processed_text"].tolist())

# 将嵌入转为DataFrame（列名：embed_0 ~ embed_383）
embed_cols = [f"embed_{i}" for i in range(text_embeddings.shape[1])]
embeddings_df = pd.DataFrame(text_embeddings, columns=embed_cols)

# 合并嵌入特征到原数据框
df = pd.concat([df, embeddings_df], axis=1)

# 查看嵌入特征维度
print(f"\nMiniLM嵌入特征维度：{text_embeddings.shape}")  # 输出：(数据行数, 384)


# - firm列（公司名）转为标签编码（若有多个公司需此步骤）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["firm_encoded"] = le.fit_transform(df["firm"])
# - year列保留数值（时间特征）

# 2. 选择核心特征列
core_feature_cols = [
    "firm_encoded", "year",  # 基础公司/时间特征
    "esg_total", "e_score", "s_score", "g_score",  # ESG词典得分
] + embed_cols  # 章节独热编码 + MiniLM嵌入

# 3. 构建最终特征矩阵
feature_matrix = df[core_feature_cols].copy()

# 4. 处理缺失值（双重保险，避免模型训练报错）
feature_matrix = feature_matrix.fillna(0)

# 5. 保存特征矩阵（用于后续模型训练）
feature_matrix.to_csv("/home/featurize/work/esg/feature.csv", index=False, encoding="utf-8")

# 输出特征矩阵信息
print(f"\n最终特征矩阵形状：{feature_matrix.shape}")  # 示例：(18, 391) = 2+4+2+384
print("\n特征矩阵前5列示例：")
print(feature_matrix[core_feature_cols[:5]].head())