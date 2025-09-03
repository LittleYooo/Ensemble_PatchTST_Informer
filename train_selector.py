import torch
import numpy as np
import os
from src.data.utils import *

import warnings
from torch.utils.data import DataLoader, TensorDataset


# 安全警告处理
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only.*")

def save_model(model, path):
    """带完整性检查的模型保存"""
    torch.save(model.state_dict(), path)
    # 验证保存
    try:
        _ = torch.load(path, weights_only=True)
        print(f"模型验证保存成功: {path}")
    except Exception as e:
        print(f"模型验证失败: {str(e)}")
        os.remove(path)  # 删除无效文件
        raise

def train():
    # 初始化环境
    os.makedirs("./saved_models", exist_ok=True)
    torch.manual_seed(42)
    
    # 生成训练数据
    predA, predB, trues = get_predictions(flag='val')
    N, L, D = predA.shape  # N: 样本数, L: 预测长度, D: 特征维度

    labels = []
    for i in range(N):
        mse_dim_vals_A = metric(predA[i:i+1], trues[i:i+1])
        mse_dim_vals_B = metric(predB[i:i+1], trues[i:i+1])
        label = (np.array(mse_dim_vals_A) < np.array(mse_dim_vals_B)).astype(float)
        labels.append(label)
    labels = np.array(labels)  # (N, D)

    features = np.concatenate([predA, predB], axis=2)  # (N, L, 2D)
    features = features.reshape(N, -1)  # (N, L * 2D)

#=========================================================================================#

    selector = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 9),  # 输出层与特征维度一致
        torch.nn.Sigmoid()
    )
    
    # 训练配置
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练循环
    for epoch in range(100):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()

            outputs = selector(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.8f}")
    
    # 保存模型（带验证）
    save_model(selector, "saved_models/selector.pth")
    
    # 测试加载
    try:
        test_model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 9),
            torch.nn.Sigmoid()
        )
        test_model.load_state_dict(
            torch.load("saved_models/selector.pth", weights_only=True)
        )
        print("模型训练&加载测试通过！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

if __name__ == "__main__":
    train()
