import numpy as np
import pandas as pd
import torch
from src.data.utils import *
import os
from config import selector_patch_len as patch_len, THRESHOLD, SAVED_MODELS_DIR
from src.models.Selector.Selector import Selector as Selector

def save_results(results):
    path = f"./results/{MODE}/{DATASET_SIZE}/{DATASET}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    average = f"{np.mean(results):.6f}"
    results = [f"{v:.6f}" for v in results]
    df = pd.DataFrame({
        "mse": results,
        "average": [average] + [""] * (len(results) - 1)
    })
    df.to_csv(path, index=False)
    print(f"结果已保存到 {path}")



def print_report(mse_A, mse_B, mse_ensemble):
    print("\n性能对比报告:")
    print(f"数据集: {DATASET}")
    print(f"模式: {MODE}")
    print("A模型MSE:", np.round(mse_A, 6)[-3:], f"last_3: {np.mean(mse_A[-3:]):.8f}")
    print("B模型MSE:", np.round(mse_B, 6)[-3:], f"last_3: {np.mean(mse_B[-3:]):.8f}")
    print("集成后MSE:", np.round(mse_ensemble, 6)[-3:], f"last_3: {np.mean(mse_ensemble[-3:]):.8f}")

    last_3_A = np.mean(mse_A[-3:])
    last_3_B = np.mean(mse_B[-3:])
    last_3_final = np.mean(mse_ensemble[-3:])

    last_3_improvement = (min(last_3_A, last_3_B) - last_3_final) / min(last_3_A, last_3_B) * 100

    print(f"last_3_improvement: {last_3_improvement:.2f}%")
    
    # save_results(mse_ensemble)

    return last_3_improvement

def test(args):
    torch.manual_seed(42)
    
    global DATASET
    DATASET = args.dset
    global DATASET_SIZE
    DATASET_SIZE = args.dset_size
    global MODE
    MODE = args.ensemble_mode

    # 生成测试数据
    predA, predB, trues = get_predictions(args, flag='test')
    N, L, D = predA.shape  # N: 样本数, L: 预测长度, D: 特征维度

    predA = predA[:, :, -3: ]  # 只取后三维
    predB = predB[:, :, -3: ]
    trues = trues[:, :, -3: ]
    N, L, D = predA.shape  # N: 样本数,

    diff = (predA - predB) ** 2
    features = np.concatenate([predA, predB, diff], axis=2)  # (N, L, 3D)
    features = features.reshape(N * L // patch_len, patch_len * 3 * D)  # (N * L // patch_len, patch_len * 3D)

    if args.ensemble_mode == 'selection':
        print("测试selection模式的选择器")
        output_dim = D
    elif args.ensemble_mode == 'stacking':
        print("测试stacking模式的选择器")
        output_dim = patch_len * D


    # 加载选择器
    selector = Selector(
        input_dim = features.shape[1],
        output_dim=output_dim,
        mode = args.ensemble_mode
    )
    path = os.path.join(SAVED_MODELS_DIR, "selector", args.ensemble_mode, f"{DATASET}.pth")
    selector.load_state_dict(torch.load(path, weights_only=True))
    
    selector.eval()

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        out_weights = selector(features_tensor)

        if args.ensemble_mode == 'selection':

            out_weights = (out_weights > THRESHOLD).float()  # (N * L // patch_len, D)
            # print("out_weights shape:", out_weights.shape)  # (N * L // patch_len, D)
            out_weights = out_weights.reshape(N, L // patch_len, D)
            out_weights = out_weights.repeat_interleave(patch_len, dim=1)

            pred_ensemble = out_weights * torch.FloatTensor(predA) + (1 - out_weights) * torch.FloatTensor(predB)
            pred_ensemble = pred_ensemble.numpy()

        elif args.ensemble_mode == 'stacking':
            pred_ensemble = out_weights.numpy().reshape(N, L, D)
            

    mse_dims_vals_A = metric(predA, trues)
    mse_dims_vals_B = metric(predB, trues)
    mse_dims_vals_ensemble = metric(pred_ensemble, trues)

    save_results(mse_on_epochs(pred_ensemble, trues))


    return print_report(mse_dims_vals_A, mse_dims_vals_B, mse_dims_vals_ensemble)

if __name__ == "__main__":
    test()
    
   

