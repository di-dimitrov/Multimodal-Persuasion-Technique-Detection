#! /bin/bash


source activate /your_path/envs/transformer_4_20

HOME_DIR="$PWD"

task_dir="task1"

train="data/task1_train.json"
dev="data/task1_dev.json"
tst="data/task1_dev.json"
model_name="bert-base-uncased"

fname=`basename $train .json`
tst_fname=`basename $tst .json`



seed=42
output_file=$HOME_DIR/results/$model_name/$tst_fname"_"$run".json"
mkdir -p $HOME_DIR/results/$model_name/$fname/

export CACHE_DIR=$HOME_DIR/cache/
export TASK_NAME="multiclass"


outputdir=$HOME_DIR/experiments/$model_name/$fname/
data_dir=$CACHE_DIR/$model_name/$fname/
mkdir -p $outputdir
mkdir -p $data_dir
echo "Output dir: "$outputdir
echo "Model Name: "$model_name
echo "Pre-trained model cache dir: "$model_cache_dir

python bin/run_glue.py \
	--model_name_or_path $model_name \
	--train_file $train \
	--validation_file $dev \
	--test_file $tst \
	--out_file $output_file \
	--do_train \
	--do_eval \
	--do_predict \
	--data_dir $data_dir \
	--max_seq_length 512 \
	--per_device_eval_batch_size=4 \
	--per_device_train_batch_size=4 \
	--learning_rate 2e-5 \
	--weight_decay 0.001 \
	--num_train_epochs 10 \
	--seed $seed \
	--output_dir $outputdir \
	--overwrite_output_dir \
	--cache_dir $data_dir \
	--load_best_model_at_end \
	--save_strategy "epoch" \
	--evaluation_strategy epoch \
	--load_best_model_at_end=True \
	--metric_for_best_model "eval_accuracy" \
	--greater_is_better=True \
	--disable_tqdm=False \
	--save_total_limit 2 \
	--do_train \
	--do_eval \
	--do_predict \
	--logging_strategy "steps"




