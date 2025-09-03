import numpy as np
from scipy.stats import wasserstein_distance

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def compute_wasserstein_distance(predictions, labels):
    """计算预测和真实值之间的Wasserstein距离"""
    w_distances = []

    # 对每个维度分别计算
    for dim in range(labels.shape[1]):
        w_dist = wasserstein_distance(predictions[:, dim], labels[:, dim])
        w_distances.append(w_dist)

    # 计算总体Wasserstein距离(所有维度的平均)
    avg_w_dist = np.mean(w_distances)

    return avg_w_dist, w_distances

def metric(pred, true, flag=0):
    
# calculate Wasserstein distance
    avg_w_dist, dim_w_dist = compute_wasserstein_distance(
        pred.reshape(-1, pred.shape[-1]),
        true.reshape(-1, pred.shape[-1])
        )
    if flag == 1:
        return avg_w_dist, dim_w_dist

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    mse_dims_vals = []
    for dim in range(true.shape[-1]):
        mse_dims_val = MSE(true[:,:,dim], pred[:,:,dim])
        mse_dims_vals.append(mse_dims_val)

    return mae, mse, mse_dims_vals, avg_w_dist, dim_w_dist
