#!/bin/bash

# This is your argument
data_path=$1
kmer=$2

seed = 1

echo "The provided kmer is: $kmer, data_path is $data_path"

# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6



for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/tf/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_tf_${data}_seed${seed} \
            --model_max_length 110 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in prom_core_all prom_core_notata
    do
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/dnabert1_${kmer} \
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
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/dnabert1_${kmer} \
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
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    for data in reconstructed
    do
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/splice/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_splice_${data}_seed${seed} \
            --model_max_length 410 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    