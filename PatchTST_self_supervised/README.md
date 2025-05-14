# trajectory-on-PatchTST

- pre-train script

```sh
python patchtst_pretrain.py --dset dim1
```

- fine-tuning script

```sh
python patchtst_finetune.py --dset dim1 --is_finetune 1 --pretrained_model <module_name>

# example
python patchtst_finetune.py --dset dim1 --is_finetune 1 --pretrained_mode saved_models/dim1/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
```

- test script

```sh
python patchtst_finetune.py --dset dim1 --pretrained_model <module_name>
```