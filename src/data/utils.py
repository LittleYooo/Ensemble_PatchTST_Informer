import numpy as np
from src.models.PatchTST.patchtst_finetune import test_func as modelB_test
from exp.exp_main import Exp_Main
import numpy as np
from types import SimpleNamespace
from config import *
import torch
import os


def get_predictions(flag = 'test'):
    cached_path = f"./cached/{DATASET}_{flag}.npz"

    if os.path.exists(cached_path) and flag == 'test':
        print(f"load predictions from cache: {cached_path}")
        data = np.load(cached_path)
        return data['predA'], data['predB'], data['trues']

    print("="*50)
    print(f"loading model A: {MODEL_A_DIR}/{MODEL_A}")
    exp = Exp_Main(argsA)
    predA, trues, mse_dim_vals_A = exp.test(setting="", test=1, data_flag=flag)  # informer
    print(mse_dim_vals_A)
    print(f"predA shape: {predA.shape}")

    torch.cuda.empty_cache()
    print("="*50)

    print(f"loading model B: {MODEL_B_DIR}/{MODEL_B}")
    B_out = modelB_test(weight_path=f"{MODEL_B_DIR}/{MODEL_B}", args=argsB, flag=flag)  # patchtst
    predB = B_out[0]
    trues2 = B_out[1]
    assert np.allclose(trues, trues2), "模型A和B的真实值不匹配"
    mse_dim_vals_B = B_out[2][0].tolist()
    
    print(f"predB shape: {predB.shape}")
    torch.cuda.empty_cache()
    print("="*50)

    cache(predA, predB, trues, cached_path)

    return predA, predB, trues


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def metric(pred, true, flag=0):
    mse = MSE(pred, true)

    mse_dims_vals = []
    for dim in range(true.shape[-1]):
        mse_dims_val = MSE(true[:,:,dim], pred[:,:,dim])
        mse_dims_vals.append(mse_dims_val)

    return mse_dims_vals

def cache(predA, predB, trues, cached_path):
    os.makedirs("./cached", exist_ok=True)
    np.savez(cached_path, predA=predA, predB=predB, trues=trues)
    print(f"predictions cached to {cached_path}")