for dataset in s0.3548_m907 CAV-H HTV2
do
    python -u run.py --train --dset $dataset > logs/train/$dataset'.log'
for dataset_size in 20552 
do
    python -u run.py --test --dset $dataset --dset_size $dataset_size > logs/$dataset'_'$dataset_size'.log'
done
done