
from torch import nn
import collections
from collections import OrderedDict
import torch
import os
from datetime import timedelta

# 初始化PyTorch的分布式数据并行（DDP）训练环境
def init_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK'))  # 获取当前进程的本地GPU编号（LOCAL_RANK）
    world_size = int(os.environ.get('WORLD_SIZE'))  # 获取全局进程总数（WORLD_SIZE）
    rank = int(os.environ.get('RANK'))              # 获取当前进程全局编号（RANK）

    torch.cuda.set_device(local_rank)               # 确保每个进程独占指定GPU，避免设备冲突
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=600)
    )

# 该函数用于递归遍历PyTorch神经网络模型的所有子模块，构建模块的层级结构字典
def nested_children(m: nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
                
    return output


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unwrap_model(model):
    unwrapped_model = nested_children(model)
    unwrapped_model = flatten_dict(unwrapped_model)
    unwrapped_model = nn.Sequential(OrderedDict(unwrapped_model))
    return unwrapped_model
    