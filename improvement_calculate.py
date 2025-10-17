import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt

from config import *
import argparse
parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# Pretraining and Finetuning
parser.add_argument('--dset', type=str, default='source_domain', help='data_set name')
parser.add_argument('--dataset_size', type=str, default='custom_data', help='data_set name')
parser.add_argument('--ensemble_mode', type=str, default='selection', help='ensemble mode: selection or stacking')
args = parser.parse_args()

csv1_path = '../PatchTST_self_supervised/saved_results/'+ args.dataset_size+'/'+ args.dset +'/'+ args.dset +'_patchtst_finetuned_cw100_tw100_patch100_stride100_epochs-finetune20_model1_acc.csv'
csv2_path ='results/'+args.ensemble_mode +'/'+ args.dataset_size +'/'+ args.dset +'/'+ args.dset +'_ensemble_acc.csv'
result_offline =pd.read_csv(csv1_path)
result_online =pd.read_csv(csv2_path)

print(f"读取文件1: {csv1_path}, 数据量: {len(result_offline)}")
print(f"读取文件2: {csv2_path}, 数据量: {len(result_online)}")

# 确保两个DataFrame都有相同的文件名
# 使用文件名作为合并键
merged_df = pd.merge(result_offline, result_online, on='轨迹', suffixes=('_1', '_2'))

print(f"匹配到的共同文件数量: {len(merged_df)}")

# 计算 1 - average_value2 / average_value1
improvement_per_trajectory = 1 - merged_df['MSE_2'] / merged_df['MSE_1']
improvement_average = np.mean(improvement_per_trajectory)
print(improvement_per_trajectory)
print(improvement_average)
df_data = {
        '提升度%': improvement_per_trajectory*100,
        '平均提升度%': [improvement_average*100] + [None] * (len(improvement_per_trajectory) - 1)  # 只在第一行显示平均值
    }
pd.DataFrame(df_data).to_csv('results/'+args.ensemble_mode +'/'+ args.dataset_size +'/'+ args.dset +'/'+ args.dset + '_improvement.csv', float_format='%.6f', index=False)

 