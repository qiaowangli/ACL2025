#!/bin/bash

data_path=$1
model_path=$2
result_name=$3
lr=3e-5

seed=1

echo "The provided data_path is $data_path"


    for data in 0 1 2 3 4
    do 
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/tf/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_tf_${data}_seed${seed} \
            --model_max_length 30 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in reconstructed
    do
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/splice/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_splice_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in  prom_core_all prom_core_notata
    do
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in prom_core_tata
    do
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    

    for data in prom_300_notata prom_300_all
    do
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done



    for data in prom_300_tata
    do 
        python /home/roy/Desktop/thesis/thesis/DNABERT_2/finetune/train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --model_path $model_path \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir $result_name/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done 


