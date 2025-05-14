# trajectory-on-PatchTST

- pre-train script

```sh
python patchtst_pretrain.py --dset dim1
```

- fine-tuning script

```sh
python patchtst_finetune.py --dset dim1 --is_finetune 1 --pretrained_model <module_name>
```

- test script

```sh
python patchtst_finetune.py --dset dim1 --pretrained_model <module_name>
```