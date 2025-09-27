import numpy as np
from src.models.PatchTST.patchtst_finetune import test_func as modelB_test
from exp.exp_main import Exp_Main
import numpy as np
from types import SimpleNamespace
from config import MODEL_A_DIR, MODEL_B_DIR, DATA_PATH
import torch
import os


def get_predictions(args, flag = 'test'):

    print("="*50)

    argsA = get_argsA(args)
    print(f"loading model A: {MODEL_A_DIR}/{args.modelA}")
    exp = Exp_Main(argsA)
    predA, trues, mse_dim_vals_A = exp.test(setting="", test=1, data_flag=flag)  # informer
    print(f"predA shape: {predA.shape}")

    torch.cuda.empty_cache()
    print("="*50)

    argsB = get_argsB(args)
    print(f"loading model B: {MODEL_B_DIR}/{args.modelB}")
    B_out = modelB_test(weight_path=f"{MODEL_B_DIR}/{args.modelB}", args=argsB, flag=flag)  # patchtst
    predB = B_out[0]
    trues2 = B_out[1]
    assert np.allclose(trues, trues2), "模型A和B的真实值不匹配"
    mse_dim_vals_B = B_out[2][0].tolist()
    
    print(f"predB shape: {predB.shape}")
    torch.cuda.empty_cache()
    print("="*50)

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

def mse_on_epochs(pred, true):
    mse_epochs = []
    for i in range(pred.shape[0]):
        mse_epoch = MSE(pred[i], true[i])
        mse_epochs.append(mse_epoch)
    return mse_epochs

def cache(predA, predB, trues, cached_path):
    os.makedirs("./cached", exist_ok=True)
    np.savez(cached_path, predA=predA, predB=predB, trues=trues)
    print(f"predictions cached to {cached_path}")

def get_argsA(args):
    argsA = SimpleNamespace(
        is_training=0,
        train_only=False,
        root_path=args.root_path,
        data_path=f"{args.dset}.csv",
        model_id=args.modelA,
        model="Informer",
        data="custom",
        features="M",
        target="OT",
        freq="h",
        individualstore_true=False,
        embed_type=4,
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
        checkpoints=MODEL_A_DIR,
        seq_len=100,
        label_len=10,
        pred_len=100,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=5,
        embed="timeF",
        enc_in=9,
        dec_in=9,
        c_out=9,
        learning_rate=0.000125,
        distil=True,
        des="Exp",
        itr=1,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        test_flop=False,
        model_name=args.modelA,
    )
    return argsA

def get_argsB(args):
    argsB = SimpleNamespace(
        is_finetune=0,
        is_linear_probe=0,
        dset_finetune=args.dset,
        context_points=100,
        target_points=100,
        batch_size=64,
        num_workers=0,
        scaler='standard',
        features='M',
        patch_len=20,
        stride=20,
        revin=0,
        use_time_features=1,
        n_layers=3,
        n_heads=16,
        d_model=64,
        d_ff=256,
        dropout=0.2,
        head_dropout=0.2,
        n_epochs_finetune=20,
        lr=1e-4,
        pretrained_model="",
        finetuned_model_id=1,
        model_type='based_model',
        dataset_size=args.dset_size,
        root_path=args.root_path,
    )
    argsB.dset = argsB.dset_finetune
    return argsB