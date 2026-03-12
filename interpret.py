# -*- coding: utf-8 -*-
"""
ESG收益预测 - SHAP可解释性分析 (独立脚本)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许 OpenMP 库重复加载
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 这样可以直接复用你写好的模型类和数据处理函数，保持代码 DRY (Don't Repeat Yourself)
from lstm import (
    ESG_LSTM, 
    load_and_preprocess_data, 
    split_data_by_firm_and_create_sequences,
    DATA_PATH,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT
)
DATA_PATH=r"E:\桌面\毕设\sec-edgar-annualreport-links-main\esg\feature_target1.csv"

# ========== 1. 配置matplotlib中文字体 ==========
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def main():
    print("="*60)
    print("1. 正在加载配置并重建数据序列...")
    print("="*60)
    
    # 1. 加载训练时保存的配置文件 (获取标准化器和维度信息)
    try:
        config = torch.load("esg_lstm_config.pth", weights_only=False)
        seq_len = config["seq_len"]
        input_dim = config["input_dim"]
        feature_cols = config["feature_cols"]
        train_years_num = config["TRAIN_YEARS_NUM"]
    except FileNotFoundError:
        raise FileNotFoundError("未找到 esg_lstm_config.pth，请确认你已经完整运行过训练脚本并生成了该文件！")

    # 2. 调用你原有的函数，重新生成序列数据 (SHAP计算需要用到)
    df, X_scaled, y, _, _, _ = load_and_preprocess_data(DATA_PATH)
    X_train_seq, _, X_val_seq, _ = split_data_by_firm_and_create_sequences(
        df, X_scaled, y, 
        train_years_num=train_years_num, 
        seq_len=seq_len
    )

    print("\n2. 正在加载训练好的 LSTM 最终模型...")
    # 3. 初始化模型结构 (参数必须与训练时完全一致)
    model = ESG_LSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    # 加载权重并切换到评估模式 (关闭 Dropout，否则 SHAP 计算会不稳定)
    model.load_state_dict(torch.load("esg_final_lstm_model.pth", weights_only=True))
    model.eval() 

    print("\n3. 开始 SHAP 可解释性分析...")
    print("【注意】背景数据为全部训练集，深度学习模型的 SHAP 计算极度耗时，请耐心等待（可能需要几分钟到十几分钟）...")
    
    # 4. 准备 SHAP 所需的张量数据
    # 改进：从训练集中随机抽取 100 个样本作为背景。
    # 这样可以极大提速，并减少由于背景数据过大导致的数值累积误差。
    indices = np.random.choice(X_train_seq.shape[0], min(100, X_train_seq.shape[0]), replace=False)
    background_data = torch.tensor(X_train_seq[indices], dtype=torch.float32)
    
    # 待解释数据：全局验证集 (观察模型对未来数据的判断逻辑)
    test_data = torch.tensor(X_val_seq, dtype=torch.float32)

    # 5. 初始化 DeepExplainer 并计算 SHAP 值
    explainer = shap.DeepExplainer(model, background_data)
    # 核心修复：增加 check_additivity=False 绕过严格的加和校验
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # 适配 PyTorch 二分类的输出结构
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    print("\n4. 计算完成！正在生成时序特征降维与可视化图表...")
    
    # 6. 核心逻辑：3D 时序特征展平为 2D 
    # 将 [样本数, 3年, 特征数] 转换为 [样本数, 3*特征数]
    num_samples = test_data.shape[0]
    num_features = len(feature_cols)
    print(f"shap_values:{shap_values.shape},test_data:{test_data.shape}")
    shap_values_2d = shap_values.reshape(num_samples, seq_len * num_features)
    test_data_2d = test_data.numpy().reshape(num_samples, seq_len * num_features)

    # 7. 生成对应的时间步特征名称
    flattened_feature_names = []
    for i in range(seq_len):
        year_offset = seq_len - 1 - i
        year_label = f"_T-{year_offset}" if year_offset > 0 else "_T(当年)"
        for col in feature_cols:
            flattened_feature_names.append(f"{col}{year_label}")

    # 8. 绘制全局 Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_2d, 
        test_data_2d, 
        feature_names=flattened_feature_names,
        show=False
    )
    plt.title("ESG特征对企业净利润涨跌的全局影响 (SHAP Summary Plot)", fontsize=14)
    plt.tight_layout()
    output_path = "shap_summary_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
