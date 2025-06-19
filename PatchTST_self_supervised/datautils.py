

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'dim1', 'dim2', 'dim3'
        ]

def get_dls(params):
    
    # assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False
        
    if params.dset == 'source_domain':
        root_path = './dataset/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,      # dataset class: 数据集类
                dataset_kwargs={                # dataset key word args
                'root_path': root_path,
                'data_path': 'source_domain.csv',
                'features': params.features,    # M MS S
                'scale': True, # True           # scale 处理
                'size': size,                   # [100, 0, 100]
                'use_time_features': params.use_time_features # False
                },
                batch_size=params.batch_size,   # 32
                workers=params.num_workers,     # number of workers
                )

    elif params.dset == 'processed_data_55':
        root_path = './dataset/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'processed_data_55.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 's0.3548_m907':
        root_path = './dataset/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 's0.3548_m907.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'CAV-H':
        root_path = './dataset/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'CAV-H.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'HTV2':
        root_path = './dataset/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'HTV2.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features,
                "train_split": 0.05,
                "test_split": 0.9
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
        
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    # dls.vars = 9, dls.len = 3, dls.c = 2
    return dls

if __name__ == "__main__":
    
    class Params:
        dset= 'dim1'
        context_points= 3
        target_points= 3
        batch_size= 8
        num_workers= 0
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)

    print('dls.train.length', len(dls.train)) # 2
    print('dls.test.length', len(dls.test)) # 2
    # print('dls.valid.length', len(dls.valid)) # 2

    # print('dls.train', dls.train)

    
    for i, batch in enumerate(dls.train):
        print(batch)
        print(i, len(batch), batch[0].shape, batch[1].shape)

    print()
    print('========================')
    print()

    # for i, batch in enumerate(dls.valid):
    #     print(i, len(batch), batch[0].shape, batch[1].shape)

    # print()
    # print('========================')
    # print()

    '''
        enumerate():
            用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    '''
    for i, batch in enumerate(dls.test):
        print(batch)
        print(i, len(batch), batch[0].shape, batch[1].shape)
        
    # breakpoint()
