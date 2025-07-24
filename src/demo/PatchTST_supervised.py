from pandas import *
import torch
from PatchTST_supervised.exp.exp_main import Exp_Main
import numpy as np
import torch
from types import SimpleNamespace


args1=SimpleNamespace(
    is_training=0,
    is_finetune=0,
    #finetune_weights='Informer_new-master\checkpoints\finetuned_s0.3548_m907_96_96_Informer_custom_ftM_sl96_ll48_pl96_dm64_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0\checkpoint.pth',
    root_path='PatchTST_supervised\dataset',
    data_path='example.csv',
    model_id='finetuned_data_example1',
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
    checkpoints='PatchTST_supervised\HTV-2.pth',
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
    enc_in=9,
    dec_in=9,
    c_out=9,
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
model_01=Exp_Main

# 非训练模式（直接加载模型进行预测和测试）
ii = 0
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args1.model_id,
                                                                                            args1
                                                                                        .model,
                                                                                            args1
                                                                                        .data,
                                                                                            args1
                                                                                        .features,
                                                                                            args1
                                                                                        .seq_len,
                                                                                            args1
                                                                                        .label_len,
                                                                                            args1
                                                                                        .pred_len,
                                                                                            args1
                                                                                        .d_model,
                                                                                            args1
                                                                                        .n_heads,
                                                                                            args1
                                                                                        .e_layers,
                                                                                            args1
                                                                                        .d_layers,
                                                                                            args1
                                                                                        .d_ff,
                                                                                            args1
                                                                                        .factor,
                                                                                            args1
                                                                                        .embed,
                                                                                            args1
                                                                                        .distil,
                                                                                            args1
                                                                                        .des, ii)

exp = model_01(args1)  

# 加载微调后的模型权重
if args1.is_finetune and args1.finetune_weights is not None:
    print(f'>>>>>>>loading fine-tuned model from {args1
.finetune_weights}<<<<<<<<')
    exp.model.load_state_dict(torch.load(args1
.finetune_weights))

# 进行测试和预测
print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting, test=1)  # 测试模型

if args1.do_predict:
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, True)  # 进行预测

torch.cuda.empty_cache()