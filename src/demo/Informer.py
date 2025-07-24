from pandas import *
import torch
import numpy as np
import torch
from types import SimpleNamespace

from Informer_new_master.exp.exp_main import Exp_Main

args2=SimpleNamespace(
    is_training=0,
    is_finetune=0,
    #finetune_weights='Informer_new-master\checkpoints\finetuned_s0.3548_m907_96_96_Informer_custom_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0\checkpoint.pth',
    root_path='PatchTST_supervised\dataset',
    data_path='example.csv',
    model_id='finetuned_data_example2',
    model='Autoformer',
    data='custom',
    features='M',
    target='OT',
    freq='h',
    individualstore_true=False,
    embed_type=0, 
    moving_avg=25, 
    dropout=0.2,
    activation='gelu',
    output_attention='store_true',
    do_predict='store_true',
    num_workers=10,  
    train_epochs=20, 
    batch_size=64, 
    patience=3,
    loss='mse', 
    lradj='type1',
    use_amp=False,
    checkpoints='Informer_new-master\checkpoints\finetuned_s0.3548_m907_96_96_Informer_custom_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0\checkpoint.pth',
    seq_len=96,
    label_len=48,
    pred_len=96,
    d_model=64,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=256,
    factor=3,
    embed='timeF',
    enc_in=7,
    dec_in=7,
    c_out=7,
    learning_rate=0.00001,
    distil=True,
    des='Exp',
    itr=1,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1,2,3',
    test_flop=False
)

# 调用 load_model 函数加载模型
model_2=Exp_Main

# 非训练模式（直接加载模型进行预测和测试）
ii = 0
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args2.model_id,
                                                                                            args2.model,
                                                                                            args2.data,
                                                                                            args2.features,
                                                                                            args2.seq_len,
                                                                                            args2.label_len,
                                                                                            args2.pred_len,
                                                                                            args2.d_model,
                                                                                            args2.n_heads,
                                                                                            args2.e_layers,
                                                                                            args2.d_layers,
                                                                                            args2.d_ff,
                                                                                            args2.factor,
                                                                                            args2.embed,
                                                                                            args2.distil,
                                                                                            args2.des, ii)

exp = model_2(args2)  

# 加载微调后的模型权重
if args2.is_finetune and args2.finetune_weights is not None:
    print(f'>>>>>>>loading fine-tuned model from {args2.finetune_weights}<<<<<<<<')
    exp.model.load_state_dict(torch.load(args2.finetune_weights))

# 进行测试和预测
print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting, test=1)  # 测试模型

if args2.do_predict:
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, True)  # 进行预测

torch.cuda.empty_cache()