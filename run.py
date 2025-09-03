import os
import sys
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    
    informer_project_path = os.path.join(current_script_path, 'src', 'models', 'Informer')
    patchtst_project_path = os.path.join(current_script_path, 'src', 'models', 'PatchTST')
    if informer_project_path not in sys.path:
        print(f"Added Informer project path to sys.path: {informer_project_path}")
        sys.path.insert(0, informer_project_path)
    
except:
    print("import error")
#================================================================================#

import train_selector
import test_selector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False, help='train the selector model')
parser.add_argument('--test', action='store_true', default=False, help='test the selector model')
parser.add_argument('--dset', type=str, default='HTV2', help='dataset name')
args = parser.parse_args()

import config
config.DATASET = args.dset


if args.train:
    print("训练选择器模型...")
    train_selector.train()
elif args.test:
    print("测试选择器模型...")
    test_selector.test()