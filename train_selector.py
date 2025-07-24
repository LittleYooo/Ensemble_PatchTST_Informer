import torch
import numpy as np
from src.data.utils import *
import os
import warnings

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

def train_selector():
    # 初始化环境
    os.makedirs("./saved_models", exist_ok=True)
    torch.manual_seed(42)

    # 加载数据
    X_train, X_val = load_data()
    
    # 生成训练数据
    predA, predB = get_predictions(X_train)

    # 计算 MSE
    mse_A = ((predA - X_val)**2).mean(axis=0)  # 按特征维度计算MSE
    mse_B = ((predB - X_val)**2).mean(axis=0)
    
    # 生成特征
    features = np.concatenate([predA, predB], axis=1)
    
    # 生成选择标签
    labels = (mse_A < mse_B).astype(float)  # 直接使用特征维度的比较结果

    # 定义选择器模型
    selector = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 9),  # 输出层与特征维度一致
        torch.nn.Sigmoid()
    )
    
    # 训练配置
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    # 训练循环
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = selector(torch.FloatTensor(features))
        
        # 广播 labels 以匹配输出维度
        loss = criterion(outputs, torch.FloatTensor(labels).expand_as(outputs))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")
    
    # 保存模型（带验证）
    save_model(selector, "./saved_models/selector.pth")
    
    # 测试加载
    try:
        test_model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 9),
            torch.nn.Sigmoid()
        )
        test_model.load_state_dict(
            torch.load("./saved_models/selector.pth", weights_only=True)
        )
        print("模型训练&加载测试通过！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

if __name__ == "__main__":
    train_selector()
