mkdir -p logs
mkdir -p logs/train
mkdir -p logs/test

for mode in stacking selection
do
    mkdir -p logs/train/$mode
    mkdir -p logs/test/$mode
    for dataset in s0.3548_m907 CAV-H HTV2
    do
        python -u run.py --train --dset $dataset --ensemble_mode $mode > logs/train/$mode/$dataset.log

        for dataset_size in 20552 
        do
            python -u run.py --test --dset $dataset --dset_size $dataset_size --ensemble_mode $mode > logs/test/$mode/$dataset'_'$dataset_size.log
        done
    done
done