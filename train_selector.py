import torch
import numpy as np
import os
from src.data.utils import *

import warnings
from torch.utils.data import DataLoader, TensorDataset
from config import selector_patch_len as patch_len, SAVED_MODELS_DIR
from src.models.Selector.Selector import Selector as Selector
from utils.tools import EarlyStopping
from src.data.utils import *
import os
from config import selector_patch_len as patch_len, SAVED_MODELS_DIR
from src.models.Selector.Selector import Selector as Selector
from src.models.PatchTST.patchtst_finetune import get_model
from src.models.PatchTST.callback.core import *
from src.models.PatchTST.callback.tracking import *
from src.models.PatchTST.callback.patch_mask import *
from src.models.PatchTST.callback.transforms import *
from src.models.PatchTST.learner import Learner
from src.models.PatchTST.data.datautils import *
from glob import glob
from config import DATA_PATH

from glob import glob

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
    torch.manual_seed(42)
    
    global DATASET
    DATASET = args.dset
    global DATASET_SIZE
    DATASET_SIZE = args.dset_size
    global MODE
    MODE = args.ensemble_mode


    # ROOT_PATH =  f'./dataset/{params.dataset_size}/{params.dset}/'
    # file_paths = sorted(glob(os.path.join(args.root_path,args.dset, "*.txt")))[-200:]
    file_paths = glob(os.path.join(args.root_path,args.dset, "*.txt"))[:10]
    print(os.path.join(args.root_path,args.dset))
    results = []
    args.root_path = os.path.join(args.root_path,args.dset)
    # print(f"loading model A: {MODEL_A_DIR}/{args.modelA}")
    # argsA = get_argsA(args)
    # exp = Exp_Main(argsA)
    # print(f"loading model B: {MODEL_B_DIR}/{args.modelB}")
    vars = 9
    argsB = get_argsB(args)
    model = get_model(vars, argsB, head_type='prediction').to('cuda')
    cbs = [RevInCB(vars, denorm=True)] if argsB.revin else []
    cbs += [PatchCB(patch_len=20, stride=20)]
    learn = Learner(model, weight_path=f"{MODEL_B_DIR}/{args.modelB}"+'.pth', cbs=cbs)

    if args.ensemble_mode == 'selection':
        print("基于selection模式的集成优化......")
        # criterion = torch.nn.BCELoss()
        # output_dim = D
    elif args.ensemble_mode == 'stacking':
        print("基于stacking模式的集成优化......")
        # criterion = torch.nn.MSELoss()
        # output_dim = patch_len * D


    path = os.path.join(SAVED_MODELS_DIR, "selector", args.ensemble_mode, f"{DATASET}.pth")
    if os.path.exists(path):
        os.remove(path)

    optimizer = None
    
    for file_path in file_paths:
        print("training on file:", file_path)
        # 生成测试数据
        args.dset = os.path.splitext(os.path.basename(file_path))[0]
        argsA = get_argsA(args)
        exp = Exp_Main(argsA)
        predA, trues = exp.test(argsA, setting="", test=1, data_flag='val')  # informer
        # print(f"predA shape: {predA.shape}")

        torch.cuda.empty_cache()
        # print("="*50)

        argsB = get_argsB(args)
        # B_out = modelB_test(weight_path=f"{MODEL_B_DIR}/{args.modelB}", args=argsB, flag='test')  # patchtst
        argsB.flag = 'val'
        dls = get_dls(argsB)
        learn.dls = dls
        dl = dls.valid
        B_out  = learn.test(dl)         # out: a list of [pred, targ, score]
        
        predB = B_out[0]
        trues2 = B_out[1]
        print(trues.shape, trues2.shape)
        assert np.allclose(trues, trues2), "模型A和B的真实值不匹配"
        # mse_dim_vals_B = B_out[2][0].tolist()
        
        torch.cuda.empty_cache()

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
            criterion = torch.nn.BCELoss()
            output_dim = D
        elif args.ensemble_mode == 'stacking':
            criterion = torch.nn.MSELoss()
            output_dim = patch_len * D
            labels = trues.reshape(N * L // patch_len, patch_len * D)  # 适配stacking的输出维度

        selector = Selector(
            input_dim = features.shape[1],
            output_dim=output_dim,
            mode = args.ensemble_mode
        )

        optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
        
        dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(labels))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        if os.path.exists(path):
            selector.load_state_dict(torch.load(path, weights_only=True))
            print("加载已有模型继续训练:", path)

        # 训练循环
        early_stopping = EarlyStopping(patience=30, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
        print(optimizer.param_groups[0]['lr'])

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

                    early_stopping(avg_last_N_loss, selector, path=path)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                loss.backward()
                optimizer.step()

            if early_stopping.early_stop:
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.8f}")

        print("当前文件训练完毕，保存模型:", path)
        print("="*70)
        # save_results(mse_on_epochs(pred_ensemble, trues))


        # return print_report(mse_dims_vals_A, mse_dims_vals_B, mse_dims_vals_ensemble)

if __name__ == "__main__":
    train()
