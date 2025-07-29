import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from ..models.patchtst.model.model import get_pts_model
from ..models.Informer.exp.exp_main import Exp_Main
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler

SAVED_MODELS_DIR = "./saved_models"
DATA_PATH = "dataset/CAV-H.csv"
MODEL_A = "model_informer_CAV-H"
MODEL_B = "model_patchtst_CAV-H"


def load_data():
    data = pd.read_csv(DATA_PATH, header=0).values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # 归一化数据
    print(data[:12])
    X_train = data[:100]  # 前100步训练
    X_val = data[100:200]  # 后100步验证
    return X_train, X_val

def load_patchtst_model(path):
    model = get_pts_model(weight_path=path)
    return model.eval()

def get_informer_predictions(X):
    args = SimpleNamespace(
        is_training=0,
        train_only=False,
        root_path="PatchTST_supervised\dataset",
        data_path="HTV2.csv",
        model_id=MODEL_A,
        model="Informer",
        data="custom",
        features="M",
        target="OT",
        freq="h",
        individualstore_true=False,
        embed_type=0,
        moving_avg=25,
        dropout=0.2,
        activation="gelu",
        output_attention="store_true",
        do_predict="store_true",
        num_workers=10,
        train_epochs=20,
        batch_size=64,
        patience=3,
        loss="mse",
        lradj="type1",
        use_amp=False,
        checkpoints="saved_models",
        seq_len=100,
        label_len=48,
        pred_len=100,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=3,
        embed="timeF",
        enc_in=9,
        dec_in=9,
        c_out=9,
        learning_rate=0.00001,
        distil=True,
        des="Exp",
        itr=1,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        test_flop=False,
    )
    exp = Exp_Main(args)
    cols = ['date', 'date.1', 'date.2', 'date.3', 'date.4', 'date.5', 'OT', 'OT.1', 'OT.2']
    X = torch.FloatTensor(X[-100:])  # 最后100步作为输入
    X = pd.DataFrame(X, columns=cols)
    return exp.predict(X, True).squeeze()  # 返回预测结果

def get_predictions(X):
    predA = get_informer_predictions(X)  # informer
    modelB = load_patchtst_model(f"{SAVED_MODELS_DIR}/{MODEL_B}.pth")  # patchtst model
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X[-100:])  # 最后100步作为输入

        X_reshaped = X_tensor.unsqueeze(0).permute(0, 2, 1).unsqueeze(1)

        # X_expanded = X_reshaped.expand(1, -1, -1, -1)  # 复制到 [64,1,9,100]

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
