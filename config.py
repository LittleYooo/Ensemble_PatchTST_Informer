from types import SimpleNamespace

SAVED_MODELS_DIR = "./saved_models"
MODEL_A_DIR = f"{SAVED_MODELS_DIR}/Informer"
MODEL_B_DIR = f"{SAVED_MODELS_DIR}/PatchTST"

# DATASET = 's0.3548_m907'
# DATASET = 'CAV-H'
DATASET = 'HTV2'

DATA_PATH = f"dataset/{DATASET}.csv"
MODEL_A = DATASET
MODEL_B = DATASET

selector_patch_len = 10     # factor of 100

argsA = SimpleNamespace(
        is_training=0,
        train_only=False,
        root_path="dataset",
        data_path=f"{DATASET}.csv",
        model_id=MODEL_A,
        model="Informer",
        data="custom",
        features="M",
        target="OT",
        freq="h",
        individualstore_true=False,
        embed_type=4,
        moving_avg=25,
        dropout=0.2,
        activation="gelu",
        output_attention="store_true",
        do_predict="store_true",
        num_workers=10,
        train_epochs=20,
        batch_size=64,
        patience=3,
        loss="mse",
        lradj="type1",
        use_amp=False,
        checkpoints="saved_models",
        seq_len=100,
        label_len=50,
        pred_len=100,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=5,
        embed="timeF",
        enc_in=9,
        dec_in=9,
        c_out=9,
        learning_rate=0.000125,
        distil=True,
        des="Exp",
        itr=1,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        test_flop=False,
        model_name=MODEL_A,
    )

# use default args
argsB = SimpleNamespace(
    is_finetune=0,
    is_linear_probe=0,
    dset_finetune=DATASET,
    context_points=100,
    target_points=100,
    batch_size=64,
    num_workers=0,
    scaler='standard',
    features='M',
    patch_len=20,
    stride=20,
    revin=0,
    use_time_features=1,
    n_layers=3,
    n_heads=16,
    d_model=64,
    d_ff=256,
    dropout=0.2,
    head_dropout=0.2,
    n_epochs_finetune=20,
    lr=1e-4,
    pretrained_model="",
    finetuned_model_id=1,
    model_type='based_model',
    dataset_size='5138',
)
argsB.dset = argsB.dset_finetune