import torch
import numpy as np
import os
from src.data.utils import *

import warnings
from torch.utils.data import DataLoader, TensorDataset
from config import selector_patch_len as patch_len, SAVED_MODELS_DIR
from src.models.Selector.Selector import Selector as Selector
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

def train(args):
    # 初始化环境
    save_path = os.path.join(SAVED_MODELS_DIR, "selector", args.ensemble_mode)
    os.makedirs(save_path, exist_ok=True)
    torch.manual_seed(42)

    # 生成训练数据
    predA, predB, trues = get_predictions(args, flag='val')
    
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

    diff = (predA - predB) ** 2
    features = np.concatenate([predA, predB, diff], axis=2)  # (N, L, 3D)
    features = features.reshape(N * L // patch_len, patch_len * 3 * D)  # (N * L // patch_len, patch_len * 3D)

#=========================================================================================#

    if args.ensemble_mode == 'selection':
        print("训练selection模式的选择器")
        criterion = torch.nn.BCELoss()
        output_dim = D
    elif args.ensemble_mode == 'stacking':
        print("训练stacking模式的选择器")
        labels = trues.reshape(N * L // patch_len, patch_len * D)  # (N * L // patch_len, patch_len * D)
        criterion = torch.nn.MSELoss()
        output_dim = patch_len * D

    selector = Selector(
        input_dim = features.shape[1],
        output_dim=output_dim,
        mode = args.ensemble_mode
    )

    # 训练配置
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)

    dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练循环
    early_stopping = EarlyStopping(patience=30, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

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
                
                tmp = optimizer.param_groups[0]['lr']
                scheduler.step(avg_last_N_loss)
                if tmp != optimizer.param_groups[0]['lr']:
                    print("Learning rate:", optimizer.param_groups[0]['lr'], end=" -> ")
                    print(optimizer.param_groups[0]['lr'])

                early_stopping(avg_last_N_loss, selector, path=os.path.join(save_path, f"{args.dset}.pth"))

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
        test_model = Selector(
            input_dim = features.shape[1],
            output_dim=output_dim,
            mode = args.ensemble_mode
        )
        test_model.load_state_dict(
            torch.load(os.path.join(save_path, f"{args.dset}.pth"), weights_only=True)
        )
        print("模型训练&加载测试通过！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

if __name__ == "__main__":
    train()
