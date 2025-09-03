from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

from PatchTST_self_supervised.patchtst_supervised import *
from PatchTST_self_supervised.src.learner import load_model, Learner
from PatchTST_self_supervised.src.data.datamodule import DataLoaders
# 加载数据集（9个特征）
df = pd.read_csv("PatchTST_supervised/dataset/example.csv")
data = df.values

# 模型配置
args = SimpleNamespace(
    context_points=100,  # Number of past points to use for prediction
    target_points=100,    # Number of future points to predict
    patch_len=10,
    stride=10,
    n_layers=3,
    n_heads=16,
    d_model=64,
    d_ff=256,
    dropout=0.2,
    head_dropout=0.1,
    res_attention=False
)
model_0 = get_model(c_in=9,args = args)
if model_0 is None:
    print("❌ 模型获取失败！")
else:
    print("✅ 模型获取成功！")

# 调用 load_model 函数加载模型
model_1 = load_model(
    path='PatchTST_supervised\HTV-2.pth',
    model = model_0,
    opt=None,
    with_opt=False, 
    device='cpu', 
    #device='cuda' if torch.cuda.is_available() else 'cpu'
)
if model_1 is None:
    print("❌ 模型加载失败！")
else:
    print("✅ 模型加载成功！")

# 准备测试数据加载器
test_tensor = torch.FloatTensor(data)
test_dataset = TensorDataset(test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32)
#dls = DataLoaders(datasetCls=test_dataloader,dataset_kwargs={},batch_size=32)
# 添加损失函数
loss_func = torch.nn.MSELoss()
#  创建 Learner 实例
learner = Learner(dls=test_dataloader, 
                 model=model_1,
                 loss_func=loss_func)
# 调用 test 前检查模型
if learner.model is None:
    raise ValueError("Model is not initialized!")

# 使用 predict 函数进行预测
scores = [mean_squared_error]
_, _, mse = learner.test(test_dataset, scores=scores)
#preds = learner.test(test_dataset) #test_data can be a tensor, numpy array, dataset or dataloader

# 使用 test 函数进行测试并计算 MSE
# scores = [mean_squared_error]
# _, _, mse = learner.test(test_dataset, scores=scores)

#print("Predict MSE:", mean_squared_error(data, preds))
print("Test MSE:", mse[0])