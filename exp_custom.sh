mkdir -p logs
mkdir -p logs/train
mkdir -p logs/test

for mode in selection #stacking
do
    mkdir -p logs/train/$mode
    mkdir -p logs/test/$mode
    for i in $(seq -f "%02g" 1 15)
    do
        # 步骤1: 训练集成器
        dataset=custom_data_${i}
        python -u run.py --train --dset $dataset --ensemble_mode $mode > logs/train/$mode/$dataset.log

        for dataset_size in 'custom_data'
        do
        # 步骤2: 测试集成器
            python -u run.py --test --do_pred --dset $dataset --dset_size $dataset_size --ensemble_mode $mode > logs/test/$mode/$dataset'_'$dataset_size.log
        # 步骤3: 计算提升度
            python -u improvement_calculate.py --dset $dataset --ensemble_mode $mode
        done
    done
done