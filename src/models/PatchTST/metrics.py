
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)


if __name__ == '__main__':
    y_pred = torch.randn(3, 5, requires_grad=True)
    y_true = torch.randn(3, 5)
    
    print(mse(y_true, y_pred))
    print(rmse(y_true, y_pred))
    print(mae(y_true, y_pred))

    # True values
    y_true = [3, -0.5, 2, 7]

    # Predicted values
    y_pred = [2.5, 0.0, 2, 8]

    # Calculate R² score
    print(r2_score(y_true, y_pred))
    print(mape(y_true, y_pred))