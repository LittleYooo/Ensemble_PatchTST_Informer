for dataset in s0.3548_m907 CAV-H HTV2
do
    python -u run.py --train --dset $dataset
for dataset_size in 5138 10276 20552 41104
do
    python -u run.py --test --dset $dataset --dset_size $dataset_size > logs/$dataset'_'$dataset_size'.log'
done
done