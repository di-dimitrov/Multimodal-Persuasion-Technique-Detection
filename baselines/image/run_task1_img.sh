#!/bin/bash

source activate /your_path/envs/transformer_4_20

task_name=propaganda_binary

### resnet18
HOME_DIR="$PWD/"
model="resnet18"
task=propaganda_binary
output_dir=$WDIR"/outputs/singletask/$task/$model/"
results_dir=$WDIR"/results/singletask/$task/$model/"
results_file=$results_dir/$task".json"
exp_name=$task_name"_"$model

python $WDIR/code/src/train.py --name=$exp_name --task-name=$task_name --seed=1 \
--train=data/train_caption.tsv --dev=data/dev_caption.tsv --test=data/test_caption.tsv \
--out-file=$results_file --data-dir=$WDIR'/data/' --best-state-path=$output_dir/best.pth --fig-dir=$output_dir \
--checkpoint-dir=$output_dir --arch=$model --batch-size=32 --learning-rate=1e-5 --weight-decay=0.00001 \
--num-epochs=100 --keep-frozen=False --use-rand-augment=False --rand-augment-n=2 --rand-augment-m=9 >&log/log.$task.$model.txt&

