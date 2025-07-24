from types import SimpleNamespace
from src.models.patchtst.model.patchtst_model.patchTST import PatchTST
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

ags = SimpleNamespace(
    context_points=100,
    patch_len=100,
    stride=20,
    target_points=100,
    n_layers=3,
    n_heads=16,
    d_model=64,
    d_ff=256,
    dropout=0.2,
    head_dropout=0.2
)

def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items(): 
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:            
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model

def get_pts_model(c_in=9, args=ags, head_type="prediction", weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
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
    if weight_path: model = transfer_weights(weight_path, model,exclude_head=False)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

if __name__ == '__main__':
    weight_path = '../../../saved_models/modelHTV2.pth'
    print(ags)
    model = get_pts_model(9, args=ags, head_type="prediction", weight_path=weight_path)
    print(model)