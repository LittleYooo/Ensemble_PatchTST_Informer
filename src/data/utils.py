import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from ..models.patchtst.model.model import get_pts_model

import numpy as np

def generate_sample_data(samples=90, dim=9):
    time = np.linspace(0, 10, samples)
    data = np.column_stack([np.sin(time + i) + np.random.normal(0, 0.1, samples) 
                           for i in range(dim)])
    np.savetxt("/dataset/trajectory.csv", data, delimiter=",")
    return data

def load_data():
    data = np.loadtxt("./dataset/trajectory.csv" ,delimiter=",")
    X_train = data[:100]  # 前100步训练
    X_val = data[100:200]  # 后100步验证
    return X_train, X_val

class DummyModel(nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, input_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))
        return self.fc(x[:, -100:, :])  # 预测10步

    def save_dummy_models():
        torch.save(DummyModel().state_dict(), "./saved_models/modelA.pth")
        torch.save(DummyModel().state_dict(), "./saved_models/modelB.pth")

def load_model(path):
    model = DummyModel()
    model.load_state_dict(torch.load(path))
    return model.eval()

def load_patchtst_model(path):
    model = get_pts_model(path)
    return model.eval()

def get_predictions(X):
    modelA = load_model("./saved_models/modelA.pth") # basic model
    modelB = load_patchtst_model("./modelHTV2.pth")  # patchtst model
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X[-100:])  # 最后100步作为输入

        X_reshaped = X_tensor.unsqueeze(0).permute(0, 2, 1).unsqueeze(1)
        print(X_reshaped.shape)  # 输出: torch.Size([1, 1, 9, 100])

        predA = modelA(X_tensor).squeeze().numpy()
        predB = modelB(X_reshaped).squeeze().numpy()
    return predA, predB

def extract_features(window):
    """输入：100×9的窗口数据"""
    features = []
    for i in range(window.shape[1]):  # 每个维度单独处理
        col = window[:, i]
        # 时域特征
        features += [
            np.mean(col), np.std(col), 
            stats.skew(col),  # 偏度
            np.corrcoef(col[:-1], col[1:])[0,1]  # 自相关
        ]
        # 频域特征
        fft = np.abs(np.fft.fft(col)[:5])  # 取前5个频率分量
        features.extend(fft)
    return np.array(features)
