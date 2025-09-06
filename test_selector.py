import numpy as np
import torch
from src.data.utils import *
import os
from config import DATASET, selector_patch_len as patch_len

def save_results(results):
    path = f"./results/{DATASET}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Dataset: {DATASET}\n")
        f.write("mse:\n")
        # 纵向
        f.write("\n".join([f"{v:.8f}" for v in results]) + "\n")
    print(f"结果已保存到 {path}")



def print_report(mse_A, mse_B, mse_ensemble):
    print("\n性能对比报告:")
    print(f"数据集: {DATASET}")
    print("A模型MSE:", np.round(mse_A, 6)[-3:], f"last_3: {np.mean(mse_A[-3:]):.8f}")
    print("B模型MSE:", np.round(mse_B, 6)[-3:], f"last_3: {np.mean(mse_B[-3:]):.8f}")
    print("集成后MSE:", np.round(mse_ensemble, 6)[-3:], f"last_3: {np.mean(mse_ensemble[-3:]):.8f}")

    last_3_A = np.mean(mse_A[-3:])
    last_3_B = np.mean(mse_B[-3:])
    last_3_final = np.mean(mse_ensemble[-3:])

    last_3_improvement = (min(last_3_A, last_3_B) - last_3_final) / min(last_3_A, last_3_B) * 100

    print(f"last_3_improvement: {last_3_improvement:.2f}%")
    
    save_results(mse_ensemble)

def test():
    torch.manual_seed(42)
    
    # 生成测试数据
    predA, predB, trues = get_predictions(flag='test')
    N, L, D = predA.shape  # N: 样本数, L: 预测长度, D: 特征维度

    predA = predA[:, :, -3: ]  # 只取后三维
    predB = predB[:, :, -3: ]
    trues = trues[:, :, -3: ]
    N, L, D = predA.shape  # N: 样本数,

    diff = (predA - predB) ** 2
    features = np.concatenate([predA, predB, diff], axis=2)  # (N, L, 3D)
    features = features.reshape(N * L // patch_len, patch_len * 3 * D)  # (N * L // patch_len, patch_len * 3D)

    # 加载选择器
    selector = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, D),
        torch.nn.Sigmoid()
    )
    selector.load_state_dict(torch.load("./saved_models/selector.pth", weights_only=True))
    
    selector.eval()

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        out_weights = selector(features_tensor)
        print("out_weights shape:", out_weights.shape)  # (N * L // patch_len, D)
        out_weights = out_weights.reshape(N, L // patch_len, D)
        out_weights = out_weights.repeat_interleave(patch_len, dim=1)

        pred_ensemble = out_weights * torch.FloatTensor(predA) + (1 - out_weights) * torch.FloatTensor(predB)
        pred_ensemble = pred_ensemble.numpy()

    mse_dims_vals_A = metric(predA, trues)
    mse_dims_vals_B = metric(predB, trues)
    mse_dims_vals_ensemble = metric(pred_ensemble, trues)

    print_report(mse_dims_vals_A, mse_dims_vals_B, mse_dims_vals_ensemble)

if __name__ == "__main__":
    test()
    
   

