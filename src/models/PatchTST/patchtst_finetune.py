

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from .models.patchTST import PatchTST
from .learner import Learner, transfer_weights
from .callback.core import *
from .callback.tracking import *
from .callback.patch_mask import *
from .callback.transforms import *
from .metrics import *
from .basics import set_device
from .data.datautils import *


set_device()

def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    # print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def test_func(weight_path, args, flag='test'):
    # get dataloader
    # print(f"loading {flag} data")
    args.flag = flag
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner( model, weight_path=weight_path+'.pth', dls=dls, cbs=cbs)
    dl = dls.test if flag=='test' else dls.valid
    # print(flag, len(dl.dataset))
    out  = learn.test(dl)         # out: a list of [pred, targ, score]
    # save results
    # pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    # # save target results
    # save_format_result(out)
    return out

def predict_func(weight_path, args, flag='test'):
    # get dataloader
    model = get_model(args.vars, args, head_type='prediction').to('cuda')
    dlss = get_dls_pred(args)
    results=[]
    for file_name, dls in dlss.items():
        print(f"Processing {file_name}")
        if args.TTA_ENABLE and args.is_test==2:        
            adapter = build_adapter(args, dls, model, weight_path)
            out = adapter.adapt() 
        else:  
        # get callbacks
            cbs = [RevInCB(args.vars, denorm=True)] if args.revin else []
            cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
            learn = Learner(dls, model,cbs=cbs)
            out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse])         # out: a list of [pred, targ, score]
        # save results
        # pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse']).to_csv(args.result_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
        # save target results
        # save_format_result(out)
        average_value = float(np.array([item[0] for item in out[3]]).mean())
        results.append({
            '轨迹': file_name,
            'MSE': average_value,
        })
    args.result_path = 'saved_results/' + '/pred/'+ args.dset_finetune + '/'
    os.makedirs(args.result_path, exist_ok=True)
    pd.DataFrame(results).to_csv(args.result_path  + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    print('results saved:')
    print(args.result_path + args.save_finetuned_model + '_acc.csv')
    return out



# if __name__ == '__main__':
        
#     if args.is_finetune:
#         args.dset = args.dset_finetune
#         # Finetune
#         suggested_lr = find_lr(head_type='prediction')        
#         finetune_func(suggested_lr)        
#         print('finetune completed')
#         # Test
#         out = test_func(args.save_path+args.save_finetuned_model)         
#         print('----------- Complete! -----------')

#     elif args.is_linear_probe:
#         args.dset = args.dset_finetune
#         # Finetune
#         suggested_lr = find_lr(head_type='prediction')        
#         linear_probe_func(suggested_lr)        
#         print('finetune completed')
#         # Test
#         out = test_func(args.save_path+args.save_finetuned_model)        
#         print('----------- Complete! -----------')

#     else:
#         args.dset = args.dset_finetune
#         weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
#         # Test
#         out = test_func(weight_path)        
#         print('----------- Complete! -----------')


