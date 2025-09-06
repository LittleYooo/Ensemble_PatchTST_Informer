import torch
import numpy as np
import os
from src.data.utils import *

import warnings
from torch.utils.data import DataLoader, TensorDataset
from config import selector_patch_len as patch_len
from utils.tools import EarlyStopping

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
    
    predA = predA[:, :, -3: ]  # 只取后三维
    predB = predB[:, :, -3: ]
    trues = trues[:, :, -3: ]
    N, L, D = predA.shape  # N: 样本数, L: 预测长度, D: 特征维度

    labels = []
    for i in range(N):
        batch_i_labels = []

        for patch_start in range(0, L, patch_len):
            patch_end = min(patch_start + patch_len, L)
            mse_dim_vals_A = metric(predA[i:i+1, patch_start:patch_end], trues[i:i+1, patch_start:patch_end])
            mse_dim_vals_B = metric(predB[i:i+1, patch_start:patch_end], trues[i:i+1, patch_start:patch_end])
            label = (np.array(mse_dim_vals_A) < np.array(mse_dim_vals_B)).astype(float)
            batch_i_labels.append(label)

        batch_i_labels = np.array(batch_i_labels)  # (L // patch_len, D)
        labels.append(batch_i_labels)

    labels = np.array(labels)  # (N, L // patch_len, D)
    labels = labels.reshape(N * L // patch_len, D)  # (N * L // patch_len, D)

    features = np.concatenate([predA, predB], axis=2)  # (N, L, 2D)
    # (N, L // patch_len, patch_len * 2D)
    features = features.reshape(N * L // patch_len, patch_len * 2 * D)
    # features = features.reshape(N, -1)  # (N, L * 2D)
    # selector_input_dim = features.shape[1]

#=========================================================================================#

    selector = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, D),  # 输出层与特征维度一致
        torch.nn.Sigmoid()
    )

    # 训练配置
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练循环
    early_stopping = EarlyStopping(patience=25, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1)

    for epoch in range(200):
        last_N_loss = []
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()

            outputs = selector(batch_features)
            loss = criterion(outputs, batch_labels)
            last_N_loss.append(loss.item())
            if len(last_N_loss) > patch_len:
                avg_last_N_loss = np.mean(last_N_loss)
                last_N_loss.pop(0)
                # last_N_loss.clear()
                
                # scheduler.step(avg_last_N_loss)
                early_stopping(avg_last_N_loss, selector, "saved_models/selector.pth")

            if early_stopping.early_stop:
                print("Early stopping")
                break

            loss.backward()
            optimizer.step()

        if early_stopping.early_stop:
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.8f}")

    # 保存模型（带验证）
    # save_model(selector, "saved_models/selector.pth")

    # 测试加载
    try:
        test_model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, D),
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
