#!/bin/bash

for dataset in s0.3548_m907 HTV2 CAV-H
do

    informer_model_path='/home/xuyan/Informer_new/checkpoints/finetuned_'$dataset'_100_100_Informer_custom_ftM_sl100_ll50_pl100_dm64_nh8_el2_dl1_df256_fc5_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
    patch_tst_model_path='/home/xuyan/trajectory-prediction-on-PatchTST/PatchTST_self_supervised/saved_models/5138/'$dataset'/masked_patchtst/based_model/'$dataset'_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1.pth'
    cp $informer_model_path ./saved_models/Informer/$dataset.pth
    cp $patch_tst_model_path ./saved_models/PatchTST/$dataset.pth
done
