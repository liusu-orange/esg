# -*- coding: utf-8 -*-
"""
ESG收益预测 - LSTM模型最终版
核心特性：
1. 单公司内时序切分（同一公司历史年份训练、未来年份验证）
2. 单公司内生成LSTM序列（彻底避免跨公司序列）
3. 无时间泄露，符合金融时序建模逻辑
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# PyTorch相关
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# sklearn工具
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. 全局配置参数（根据你的数据调整） ==========
# 数据路径
DATA_PATH = r"/home/featurize/work/esg/feature_target1.csv"

# LSTM模型参数
SEQ_LEN = 3  # 单公司内连续N年生成序列（建议≤TRAIN_YEARS_NUM）
HIDDEN_DIM = 64  # LSTM隐藏层维度
NUM_LAYERS = 2  # LSTM层数
DROPOUT = 0.2  # Dropout率（防止过拟合）

# 训练参数
EPOCHS = 500  # 训练轮次
BATCH_SIZE = 16  # 批次大小（样本少可设为4/2）
LEARNING_RATE = 0.001  # 学习率

# 单公司时序切分参数
TRAIN_YEARS_NUM = 5  # 单公司用前N年训练，第N+1年验证（需保证单公司年份≥N+1）

# ========== 2. 配置matplotlib中文字体（避免中文乱码） ==========
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei"]

# ========== 3. 数据加载与预处理 ==========
def load_and_preprocess_data(data_path):
    """
    加载数据并预处理：
    1. 校验必要列
    2. 按公司+年份排序（保证单公司内年份递增）
    3. 特征标准化
    """
    # 加载数据
    df = pd.read_csv(data_path, encoding="utf-8")
    
    # 校验必要列
    required_cols = ["firm_encoded", "year", "target"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列：{missing_cols}，请检查数据文件！")
    
    # 核心排序：先按公司编码，再按年份（单公司内年份递增）
    df = df.sort_values(["firm_encoded", "year"]).reset_index(drop=True)
    
    # 分离特征和标签
    feature_cols = [col for col in df.columns if col not in required_cols]
    if len(feature_cols) == 0:
        raise ValueError("数据中无特征列！请检查数据结构")
    
    X = df[feature_cols]
    y = df["target"].values
    
    # 特征标准化（消除量纲影响）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 基础信息统计
    input_dim = X_scaled.shape[1]  # 特征维度（LSTM输入维度）
    total_firms = df["firm_encoded"].nunique()  # 总公司数
    # 统计有效公司数（年份≥TRAIN_YEARS_NUM+1）
    valid_firms = 0
    for firm in df["firm_encoded"].unique():
        firm_years = df[df["firm_encoded"] == firm]["year"].nunique()
        if firm_years >= TRAIN_YEARS_NUM + 1:
            valid_firms += 1
    
    # 打印数据信息
    print("="*50)
    print("数据加载完成！基础信息：")
    print(f"特征维度：{input_dim} | 总样本数：{X_scaled.shape[0]}")
    print(f"总公司数：{total_firms} | 有效公司数（年份≥{TRAIN_YEARS_NUM+1}）：{valid_firms}")
    print("="*50)
    
    return df, X_scaled, y, input_dim, scaler, feature_cols

# ========== 4. 单公司内生成LSTM序列（核心：无跨公司序列） ==========
def create_sequences_single_firm(X, y, seq_len=3):
    """
    仅在单公司内生成时序序列：
    - X/y：单公司的特征/标签（已按年份排序）
    - seq_len：时间步长（连续N年）
    返回：该公司的三维序列特征+一维标签
    """
    X_seq = []  # 存储序列特征：[seq1, seq2,...]，每个seq是(seq_len, input_dim)
    y_seq = []  # 存储序列标签：每个序列对应最后一年的标签
    
    # 仅当数据量≥时间步长时生成序列
    if len(X) >= seq_len:
        # 遍历生成所有可能的连续序列
        for i in range(len(X) - seq_len + 1):
            # 取连续seq_len年的特征
            X_seq.append(X[i:i+seq_len])
            # 标签为序列最后一年的收益（预测目标）
            y_seq.append(y[i+seq_len-1])
    
    
    # 转换为numpy数组（空序列返回空数组）
    X_seq_np = np.array(X_seq) if X_seq else np.array([])
    y_seq_np = np.array(y_seq) if y_seq else np.array([])
    print(f"单次转换序列维度：{X_seq_np.shape},{y_seq_np.shape}")
    return X_seq_np, y_seq_np

# ========== 5. 按公司分组切分+生成序列（核心修复：无跨公司） ==========
def split_data_by_firm_and_create_sequences(df, X_scaled, y, train_years_num=5, seq_len=3):
    """
    核心流程：
    1. 遍历每个公司
    2. 单公司内时序切分（历史年份训练，未来年份验证）
    3. 单公司内生成序列（无跨公司）
    4. 拼接所有公司的序列→全局训练/验证序列
    """
    # 存储所有公司的训练/验证序列
    X_train_seq_all = []
    y_train_seq_all = []
    X_val_seq_all = []
    y_val_seq_all = []
    
    # 记录各公司的切分年份（用于校验）
    firm_year_mapping = {}
    
    # 遍历每个公司
    for firm in df["firm_encoded"].unique():
        # 1. 提取单公司数据
        firm_mask = df["firm_encoded"] == firm
        firm_df = df[firm_mask].copy()
        firm_X = X_scaled[firm_mask]
        firm_y = y[firm_mask]
        
        # 2. 单公司年份校验
        firm_years = sorted(firm_df["year"].unique())
        if len(firm_years) < train_years_num + 1:
            continue  # 年份不足，跳过该公司
        
        # 3. 单公司内时序切分：历史训，未来验
        train_years = firm_years[:train_years_num]  # 训练年份（前N年）
        val_year = firm_years[train_years_num:]      # 验证年份（后N年）
        
        # 4. 筛选单公司的训练/验证数据（静态特征）
        train_mask = firm_df["year"].isin(train_years)
        val_mask = firm_df["year"].isin(val_year)
        
        firm_X_train = firm_X[train_mask]
        firm_y_train = firm_y[train_mask]
        firm_X_val = firm_X[val_mask]
        firm_y_val = firm_y[val_mask]
  
        # 5. 单公司内生成序列（核心：无跨公司）
        firm_X_train_seq, firm_y_train_seq = create_sequences_single_firm(firm_X_train, firm_y_train, seq_len)
        firm_X_val_seq, firm_y_val_seq = create_sequences_single_firm(firm_X_val, firm_y_val, seq_len)
        
        # 6. 收集有效序列（非空才加入全局）
        if len(firm_X_train_seq) > 0 and len(firm_X_val_seq) > 0:
            X_train_seq_all.append(firm_X_train_seq)
            y_train_seq_all.append(firm_y_train_seq)
            X_val_seq_all.append(firm_X_val_seq)
            y_val_seq_all.append(firm_y_val_seq)
            # 记录该公司的切分年份
            firm_year_mapping[firm] = {
                "train_years": train_years,
                "val_year": val_year
            }
    
    # 7. 拼接所有公司的序列→全局序列（LSTM输入格式）
    if not X_train_seq_all or not X_val_seq_all:
        raise ValueError(
            f"全局训练/验证序列为空！请降低参数：\n"
            f"- TRAIN_YEARS_NUM（当前{train_years_num}）\n"
            f"- SEQ_LEN（当前{seq_len}）"
        )
    
    X_train_seq = np.concatenate(X_train_seq_all, axis=0)
    y_train_seq = np.concatenate(y_train_seq_all, axis=0)
    X_val_seq = np.concatenate(X_val_seq_all, axis=0)
    y_val_seq = np.concatenate(y_val_seq_all, axis=0)
    
    # 打印序列信息
    print("\n序列生成完成！序列信息：")
    print(f"参与序列生成的公司数：{len(firm_year_mapping)}")
    print(f"全局训练序列数：{X_train_seq.shape[0]} | 序列维度：{X_train_seq.shape}")
    print(f"全局验证序列数：{X_val_seq.shape[0]} | 序列维度：{X_val_seq.shape}")
    # 打印前3个公司的切分示例
    sample_firms = list(firm_year_mapping.keys())[:3]
    for firm in sample_firms:
        mapping = firm_year_mapping[firm]
        print(f"公司{firm}：训练年份{mapping['train_years']} → 验证年份{mapping['val_year']}")
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq

# ========== 6. LSTM模型定义 ==========
class ESG_LSTM(nn.Module):
    """
    适配ESG收益预测的LSTM模型：
    - 输入：三维时序序列 [batch_size, seq_len, input_dim]
    - 输出：二维概率值 [batch_size, 1]（sigmoid激活，0-1之间）
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(ESG_LSTM, self).__init__()
        
        # LSTM层（核心时序建模）
        self.lstm = nn.LSTM(
            input_size=input_dim,      # 输入维度（特征数）
            hidden_size=hidden_dim,    # 隐藏层维度
            num_layers=num_layers,     # 层数
            batch_first=True,          # 输入格式：[batch_size, seq_len, input_dim]
            dropout=dropout if num_layers > 1 else 0  # 单层LSTM不使用Dropout
        )
        
        # 全连接层（将LSTM输出映射到预测概率）
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # 降维
            nn.ReLU(),                  # 激活函数（非线性）
            nn.Dropout(dropout),        # Dropout防止过拟合
            nn.Linear(32, 1),           # 最终输出：二分类概率
            nn.Sigmoid()                # 激活到0-1区间
        )
    
    def forward(self, x):
        """前向传播：输入→LSTM→全连接→输出"""
        # LSTM层输出：(batch_size, seq_len, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出（代表序列的最终特征）
        last_time_step_out = lstm_out[:, -1, :]
        # print(f"last-time:{last_time_step_out.shape}")
        # 全连接层输出概率
        output = self.fc_layers(last_time_step_out)
        # print(f"output:{output.shape}")
        return output

# ========== 7. LSTM模型训练与评估 ==========
def train_lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_dim):
    """
    训练LSTM模型：
    - 输入：全局训练/验证序列（已在单公司内生成）
    - 输出：训练好的模型+验证集准确率
    """
    # 转换为PyTorch张量（适配GPU/CPU）
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
    
    # 构建数据加载器（批次迭代）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = ESG_LSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # 优化器和损失函数（二分类）
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失（适配sigmoid输出）
    
    # 训练循环
    print("\n" + "="*90)
    print(f"开始训练LSTM模型（{EPOCHS}轮）")
    print("="*90)
    for epoch in range(EPOCHS):
        model.train()  # 训练模式（启用Dropout）
        total_loss = 0.0
        batch_count = 0
        
        # 批次训练
        for batch_X, batch_y in train_loader:
            # print(f"batch_X:{batch_X.shape},batch_y:{batch_y.shape}")
            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_X)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播（计算梯度）
            optimizer.step()  # 更新参数
            
            total_loss += loss.item()
            batch_count += 1
        
        # 每10轮打印一次损失
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch [{epoch+1}/{EPOCHS}] | 平均批次损失：{avg_loss:.4f}")
    
    # 验证集评估
    print("\n" + "="*50)
    print("模型训练完成！开始验证集评估")
    print("="*50)
    model.eval()  # 评估模式（关闭Dropout）
    with torch.no_grad():  # 禁用梯度计算（提速）
        val_outputs = model(X_val_tensor)
        # 转换为标签（概率>0.5为1，否则为0）
        y_pred = (val_outputs > 0.5).float().squeeze().numpy()
    
    # 计算准确率
    accuracy = accuracy_score(y_val_seq, y_pred)
    # 打印评估结果
    print(f"验证集准确率：{accuracy:.4f}")
    print("\n分类报告（详细评估）：")
    print(classification_report(
        y_val_seq, y_pred,
        target_names=["负收益", "正收益"],
        zero_division=0  # 避免无样本时的警告
    ))
    
    return model, accuracy

# ========== 8. 训练最终模型（全量数据） ==========
def train_final_model(df, X_scaled, y, input_dim, seq_len=3):
    """用全量数据训练最终模型（所有公司的所有年份）"""
    # 全量数据生成序列（单公司内生成）
    X_full_seq_all = []
    y_full_seq_all = []
    
    for firm in df["firm_encoded"].unique():
        firm_mask = df["firm_encoded"] == firm
        firm_X = X_scaled[firm_mask]
        firm_y = y[firm_mask]
        # 单公司内生成序列
        firm_X_seq, firm_y_seq = create_sequences_single_firm(firm_X, firm_y, seq_len)
        if len(firm_X_seq) > 0:
            X_full_seq_all.append(firm_X_seq)
            y_full_seq_all.append(firm_y_seq)
    
    if not X_full_seq_all:
        raise ValueError("全量数据生成序列为空！请降低SEQ_LEN")
    
    # 拼接全量序列
    X_full_seq = np.concatenate(X_full_seq_all, axis=0)
    y_full_seq = np.concatenate(y_full_seq_all, axis=0)
    
    # 转换张量
    X_full_tensor = torch.tensor(X_full_seq, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full_seq, dtype=torch.float32).unsqueeze(1)
    
    # 初始化最终模型
    final_model = ESG_LSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # 训练最终模型
    print("\n" + "="*50)
    print(f"开始训练最终模型（全量数据，{EPOCHS}轮）")
    print("="*50)
    final_model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = final_model(X_full_tensor)
        loss = criterion(outputs, y_full_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | 损失：{loss.item():.4f}")
    
    # 保存模型和配置
    torch.save(final_model.state_dict(), "esg_final_lstm_model.pth")
    torch.save({
        "scaler": scaler,
        "seq_len": SEQ_LEN,
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "TRAIN_YEARS_NUM": TRAIN_YEARS_NUM
    }, "esg_lstm_config.pth")
    
    print("\n模型保存完成！")
    print("最终模型文件：esg_final_lstm_model.pth")
    print("配置文件（含标准化器）：esg_lstm_config.pth")
    
    return final_model

# ========== 9. 主函数（执行全流程） ==========
if __name__ == "__main__":
    try:
        # 1. 加载并预处理数据
        df, X_scaled, y, input_dim, scaler, feature_cols = load_and_preprocess_data(DATA_PATH)
        
        # 2. 按公司切分+生成序列（核心：无跨公司序列）
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = split_data_by_firm_and_create_sequences(
            df, X_scaled, y,
            train_years_num=TRAIN_YEARS_NUM,
            seq_len=SEQ_LEN
        )
        
        # 3. 训练LSTM模型并评估
        lstm_model, val_accuracy = train_lstm_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            input_dim=input_dim
        )
        
        # 4. 训练最终模型（全量数据）
        final_model = train_final_model(
            df, X_scaled, y,
            input_dim=input_dim,
            seq_len=SEQ_LEN
        )
        
        print("\n" + "="*50)
        print("全流程执行完成！")
        print(f"验证集准确率：{val_accuracy:.4f}")
        print("模型文件已保存至当前目录！")
        print("="*50)
    
    except Exception as e:
        print(f"\n程序执行出错：{e}")
        print("请根据错误信息检查数据或参数配置！")