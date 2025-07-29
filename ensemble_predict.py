import numpy as np
import torch
from src.data.utils import *
import matplotlib.pyplot as plt

def evaluate_models(X_train, X_val):
    """评估单个模型性能"""
    predA, predB = get_predictions(X_train)
    
    # 计算各维度MSE
    mse_A = ((predA - X_val)**2).mean(axis=0)
    mse_B = ((predB - X_val)**2).mean(axis=0)
    last_3_A = mse_A[-3:].mean()
    last_3_B = mse_B[-3:].mean()
    
    print("\n" + "="*50)
    print(f"ModelA 各维度MSE: {np.round(mse_A, 4)} | 平均: {mse_A.mean():.4f} | last_3: {np.round(last_3_A, 4)}")
    print(f"ModelB 各维度MSE: {np.round(mse_B, 4)} | 平均: {mse_B.mean():.4f} | last_3: {np.round(last_3_B, 4)}")
    print("="*50)
    
    return predA, predB, mse_A.mean(), mse_B.mean(), last_3_A, last_3_B

def dynamic_ensemble_predict():
    # 加载数据
    X_train, X_val = load_data()
    
    # 评估基础模型
    predA, predB, mseA, mseB, last_3_A, last_3_B = evaluate_models(X_train, X_val)
    
    # 生成特征
    features_val = np.concatenate([predA, predB], axis=1)  # 生成特征
    
    # 加载选择器
    selector = torch.nn.Sequential(
        torch.nn.Linear(18, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 9),
        torch.nn.Sigmoid()
    )
    selector.load_state_dict(torch.load("./saved_models/selector.pth", weights_only=True))
    
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_val)  # 不需要添加批次维度
        selection_probs = selector(features_tensor).numpy()  # 预测选择概率

    # 动态选择，确保 choices 与 predA 和 predB 形状一致
    choices = selection_probs > 0.5  
    #choices = choices.astype(np.float32)  # 转换为 float 类型以便后续计算

    # 集成预测
    final_pred = choices * predA + (1-choices) * predB  # 使用广播机制
    mse_ensemble = ((final_pred - X_val)**2).mean()

    mse_final = ((final_pred - X_val)**2).mean(axis=0)
    last_3_final = mse_final[-3:].mean()
    print(f"集成后的各维度MSE: {np.round(mse_final, 4)} | 平均: {mse_final.mean():.4f} | last_3: {np.round(last_3_final, 4)}")
    print("="*50 + "\n")
    # 性能对比报告
    print("\n性能对比报告:")
    print(f"ModelA 平均MSE: {mseA:.4f}")
    print(f"ModelB 平均MSE: {mseB:.4f}")
    print(f"动态集成MSE: {mse_ensemble:.4f}")
    improvement = (min(mseA, mseB) - mse_ensemble) / min(mseA, mseB) * 100
    last_3_improvement = (min(last_3_A, last_3_B) - last_3_final) / min(last_3_A, last_3_B) * 100
    print(f"相对最佳模型提升: {improvement:.2f}% | last_3提升: {last_3_improvement:.2f}%")
    
   

if __name__ == "__main__":
    dynamic_ensemble_predict()
    
   

