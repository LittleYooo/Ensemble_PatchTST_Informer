# trajectory-on-PatchTST

- pre-train script

```sh
python patchtst_pretrain.py --dset source_domain
```

- linear-probe script

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --pretrained_model <module_path>

# examples
python patchtst_finetune.py --dset CAV-H --is_linear_probe 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset s0.3548_m907 --is_linear_probe 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset HTV2 --is_linear_probe 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset processed_data_55 --is_linear_probe 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
```

- fine-tuning script

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --pretrained_model <module_path>

# examples
python patchtst_finetune.py --dset CAV-H --is_finetune 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset s0.3548_m907 --is_finetune 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset HTV2 --is_finetune 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset processed_data_55 --is_finetune 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
```

